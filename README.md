# CS336 Lecture Assistant (RAG)

A Retrieval-Augmented Generation assistant over the Stanford CS336
"Language Modeling from Scratch" lecture series.

> Work in progress — full documentation lands with the final release.

## Development

Run the same checks as CI:

```bash
make ci
```

Useful individual targets:

```bash
make lint
make format-check
make typecheck
make test
make pre-commit
```

Install local pre-commit hooks:

```bash
make pre-commit-install
```
