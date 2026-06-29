# policy/

Trained policy weights are **not** included in this repository (commissioned research output).
The full observation/action interface is specified in [`../docs/POLICY_IO.md`](../docs/POLICY_IO.md),
so the deployment stack is complete and legible without them.

To run the stack, place a compatible 29-DoF policy here (`*.onnx`, obs 1432 / action 29) or
point the `IGRIS_POLICY` environment variable at one.

> `*.onnx` is git-ignored by default. If a specific evaluation policy is cleared for release,
> add it explicitly with `git add -f policy/<name>.onnx`.
