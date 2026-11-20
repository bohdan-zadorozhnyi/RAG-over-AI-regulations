"""Build a small index in CI without network access or a GPU.

Checked-in excerpts plus a deterministic fake embedder, so the eval measures
retrieval and fusion logic rather than model quality. Keep it under a second.
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
