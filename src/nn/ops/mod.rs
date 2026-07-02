//! Shared building blocks for graph construction.
//!
//! Stage 1 of the refactor (see REFACTOR.md): `meta` is the single source of
//! truth for every kernel's Meta-buffer layout. Stage 2 adds one node-emitter
//! per op here, shared by the train / prefill / decode graph builders.

pub mod meta;
