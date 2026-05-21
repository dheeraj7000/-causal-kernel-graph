//===-- CagPassInstrumentation.cpp - CAG MLIR pass instrumentation --------===//
#include "CagPassInstrumentation.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

using namespace mlir;
using cag::CagPassInstrumentation;

namespace {

// JSON string escape: handle quotes, backslashes, newlines, control chars.
std::string jsonEscape(StringRef s) {
  std::string out;
  out.reserve(s.size() + 8);
  out.push_back('"');
  for (char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(c);
        }
    }
  }
  out.push_back('"');
  return out;
}

// Format unsigned int.
std::string u64(uint64_t v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%llu", (unsigned long long)v);
  return buf;
}

} // namespace

namespace cag {

CagPassInstrumentation::CagPassInstrumentation(FILE *sink, bool ownsSink)
    : sink_(sink), ownsSink_(ownsSink) {}

CagPassInstrumentation::~CagPassInstrumentation() {
  if (ownsSink_ && sink_) {
    std::fflush(sink_);
    std::fclose(sink_);
  }
}

std::unique_ptr<CagPassInstrumentation> CagPassInstrumentation::fromEnv() {
  // Prefer an inherited FD (from a parent process); fall back to path.
  if (const char *fdStr = std::getenv("TRITON_CAG_EVENTS_FD")) {
    int fd = std::atoi(fdStr);
    if (fd > 0) {
      FILE *fp = ::fdopen(fd, "a");
      if (fp) {
        std::setvbuf(fp, nullptr, _IOLBF, 0);
        return std::make_unique<CagPassInstrumentation>(fp, /*owns=*/false);
      }
    }
  }
  if (const char *path = std::getenv("TRITON_CAG_EVENTS_PATH")) {
    FILE *fp = std::fopen(path, "a");
    if (fp) {
      std::setvbuf(fp, nullptr, _IOLBF, 0);
      return std::make_unique<CagPassInstrumentation>(fp, /*owns=*/true);
    }
  }
  return nullptr;
}

uint64_t CagPassInstrumentation::readUID(Operation *op) {
  if (auto attr = op->getAttrOfType<StringAttr>(kUidAttrName)) {
    try {
      return std::stoull(attr.getValue().str());
    } catch (...) {
      return 0;
    }
  }
  return 0;
}

uint64_t CagPassInstrumentation::readParentUID(Operation *op) {
  Operation *parent = op->getParentOp();
  return parent ? readUID(parent) : 0;
}

std::string CagPassInstrumentation::renderLoc(Operation *op) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  op->getLoc().print(os);
  os.flush();
  return buf;
}

uint64_t CagPassInstrumentation::tagSubtree(Operation *root) {
  uint64_t newlyTagged = 0;
  // walk() visits root and all nested ops (depth-first, pre-order by default).
  root->walk([&](Operation *op) {
    if (!op->hasAttr(kUidAttrName)) {
      uint64_t uid = uidFactory_.next();
      op->setAttr(kUidAttrName,
                  StringAttr::get(op->getContext(), std::to_string(uid)));
      ++newlyTagged;
    }
  });
  return newlyTagged;
}

void CagPassInstrumentation::emit(const std::string &line) {
  std::lock_guard<std::mutex> lock(sinkMutex_);
  std::fputs(line.c_str(), sink_);
  std::fputc('\n', sink_);
}

void CagPassInstrumentation::runBeforePass(Pass *pass, Operation *op) {
  uint64_t passIdx = passSeq_++;
  uint64_t newTags = tagSubtree(op);

  // Header record: pass entry summary.
  std::ostringstream hdr;
  hdr << "{"
      << "\"type\":\"pass_before_cpp\","
      << "\"pass_idx\":" << passIdx << ","
      << "\"pass_name\":" << jsonEscape(pass->getName()) << ","
      << "\"pass_arg\":" << jsonEscape(pass->getArgument()) << ","
      << "\"root_op\":" << jsonEscape(op->getName().getStringRef()) << ","
      << "\"root_uid\":" << u64(readUID(op)) << ","
      << "\"newly_tagged\":" << newTags
      << "}";
  emit(hdr.str());

  // Per-op records: emit a row for each op currently live under the root.
  op->walk([&](Operation *o) {
    uint64_t uid = readUID(o);
    if (!uid) return; // shouldn't happen; we just tagged
    std::ostringstream row;
    row << "{"
        << "\"type\":\"op_before_cpp\","
        << "\"pass_idx\":" << passIdx << ","
        << "\"uid\":" << u64(uid) << ","
        << "\"parent_uid\":" << u64(readParentUID(o)) << ","
        << "\"op_name\":" << jsonEscape(o->getName().getStringRef()) << ","
        << "\"loc\":" << jsonEscape(renderLoc(o)) << ","
        << "\"num_operands\":" << o->getNumOperands() << ","
        << "\"num_results\":" << o->getNumResults()
        << "}";
    emit(row.str());
  });
}

void CagPassInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  // Note: passSeq_ already advanced in runBeforePass; the "after" record uses
  // (passSeq_ - 1) to refer to the pass we are leaving.
  uint64_t passIdx = passSeq_ - 1;
  uint64_t newTags = tagSubtree(op);  // new ops created during the pass

  std::ostringstream hdr;
  hdr << "{"
      << "\"type\":\"pass_after_cpp\","
      << "\"pass_idx\":" << passIdx << ","
      << "\"pass_name\":" << jsonEscape(pass->getName()) << ","
      << "\"newly_tagged\":" << newTags
      << "}";
  emit(hdr.str());

  op->walk([&](Operation *o) {
    uint64_t uid = readUID(o);
    if (!uid) return;
    std::ostringstream row;
    row << "{"
        << "\"type\":\"op_after_cpp\","
        << "\"pass_idx\":" << passIdx << ","
        << "\"uid\":" << u64(uid) << ","
        << "\"parent_uid\":" << u64(readParentUID(o)) << ","
        << "\"op_name\":" << jsonEscape(o->getName().getStringRef()) << ","
        << "\"loc\":" << jsonEscape(renderLoc(o))
        << "}";
    emit(row.str());
  });
}

void CagPassInstrumentation::runAfterPassFailed(Pass *pass, Operation *op) {
  uint64_t passIdx = passSeq_ - 1;
  std::ostringstream rec;
  rec << "{"
      << "\"type\":\"pass_failed_cpp\","
      << "\"pass_idx\":" << passIdx << ","
      << "\"pass_name\":" << jsonEscape(pass->getName())
      << "}";
  emit(rec.str());
}

} // namespace cag

//===----------------------------------------------------------------------===//
// Triton integration entry point
//
// Triton's CUDABackend has a static `instrumentation` field that, if set,
// receives calls .load_dialects(ctx) and .patch(stage_name, pm, ctx). This
// shared library exposes a C symbol `cag_register_instrumentation(PM)` that
// callers can invoke after constructing a pass manager to attach this
// PassInstrumentation. We also provide a constructor symbol that runs on
// dlopen, registering with all subsequent PassManagers via a global hook is
// not officially supported in upstream MLIR, so the recommended path is to
// patch Triton's `nvidia/compiler.py` to call us. See BUILD.md.
//===----------------------------------------------------------------------===//
#include "mlir/Pass/PassManager.h"

extern "C" {

/// Attach a CagPassInstrumentation to the given PassManager. Returns 1 if
/// attached, 0 if no sink is configured via env (silent no-op).
int cag_attach_to_pass_manager(mlir::PassManager *pm) {
  auto inst = cag::CagPassInstrumentation::fromEnv();
  if (!inst) return 0;
  pm->addInstrumentation(std::move(inst));
  return 1;
}

} // extern "C"
