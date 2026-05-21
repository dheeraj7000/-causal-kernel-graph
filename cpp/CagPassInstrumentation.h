//===-- CagPassInstrumentation.h - CAG MLIR pass instrumentation -*- C++ -*-=//
//
// PassInstrumentation that gives every MLIR operation a stable UID, then
// records every pass entry/exit to an NDJSON event stream. This is the C++
// backbone for the Causal Attribution Graph's IRNode-level cross-pass identity
// (the "transformed_to: IRNode -> IRNode" edges).
//
// Design notes
// ------------
// MLIR's PassInstrumentation interface gives us:
//   * runBeforePass(Pass *, Operation *)
//   * runAfterPass (Pass *, Operation *)
//   * runBeforeAnalysis / runAfterAnalysis
//   * runAfterPassFailed
// These callbacks receive the *root* operation that the pass is running on
// (typically a ModuleOp or a FuncOp). Operation* pointers are NOT stable
// across passes -- a pass may erase and re-create operations. To recover
// identity we tag each op with a NamedAttribute "cag.uid" the first time we
// see it; MLIR's clone/copy paths propagate Attributes verbatim, so the same
// op surviving a pass keeps the same UID, and a newly created op simply
// doesn't have one (we assign on next sweep).
//
// We *do not* augment Locations -- locations get rewritten in passes that
// fuse/inline and can lose track. NamedAttributes survive standard clones.
//
// The event sink is a process-wide file descriptor configured by the env
// var TRITON_CAG_EVENTS_FD (preferred for in-process attachment) or, if
// unset, TRITON_CAG_EVENTS_PATH (a file path the instrumentation opens
// itself with O_APPEND). One JSON object per line.
//
//===----------------------------------------------------------------------===//
#ifndef CAG_PASS_INSTRUMENTATION_H
#define CAG_PASS_INSTRUMENTATION_H

#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>

namespace cag {

/// Stable UID counter. Process-wide, atomic, monotonically increasing.
/// We could content-hash instead, but for cross-pass identity what matters is
/// that the same Operation* keeps the same UID, which a monotonic counter
/// trivially guarantees.
class UIDFactory {
public:
  uint64_t next() { return counter_.fetch_add(1, std::memory_order_relaxed); }
private:
  std::atomic<uint64_t> counter_{1};
};

/// PassInstrumentation that:
///   1. On runBeforePass: walks the op (and its region), tagging any untagged
///      operation with a fresh "cag.uid" StringAttr.
///   2. Emits an NDJSON record for the pass entry (uids of all live ops).
///   3. On runAfterPass: walks again, emits records for ops still alive
///      (transformed_to source UID == target UID; the op survived) and ops
///      newly created (no prior UID until this pass tagged them).
///   4. Locks the sink with a mutex; thread-safe for concurrent passes.
class CagPassInstrumentation : public mlir::PassInstrumentation {
public:
  /// Construct from an opened FILE* (line-buffered). Takes ownership iff
  /// `ownsSink` is true.
  CagPassInstrumentation(FILE *sink, bool ownsSink);

  /// Construct from env vars (TRITON_CAG_EVENTS_FD / TRITON_CAG_EVENTS_PATH).
  /// Returns nullptr if no sink is configured (do not register).
  static std::unique_ptr<CagPassInstrumentation> fromEnv();

  ~CagPassInstrumentation() override;

  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override;
  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override;
  void runAfterPassFailed(mlir::Pass *pass, mlir::Operation *op) override;

private:
  /// Ensure every op in the subtree rooted at `root` has a "cag.uid"
  /// StringAttr. Returns the count of ops newly tagged in this sweep.
  uint64_t tagSubtree(mlir::Operation *root);

  /// Emit one NDJSON record (newline-terminated).
  void emit(const std::string &line);

  /// Read the current UID attribute, or 0 if missing.
  static uint64_t readUID(mlir::Operation *op);

  /// Read parent op's UID (0 if root / parent missing).
  static uint64_t readParentUID(mlir::Operation *op);

  /// Render an op's location as "file:line:col" or "unknown".
  static std::string renderLoc(mlir::Operation *op);

  FILE *sink_;
  bool ownsSink_;
  std::mutex sinkMutex_;
  UIDFactory uidFactory_;
  uint64_t passSeq_{0};

  static constexpr const char *kUidAttrName = "cag.uid";
};

} // namespace cag

#endif // CAG_PASS_INSTRUMENTATION_H
