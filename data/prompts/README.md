# Prompts

`gutenberg_300k.txt` is the text the kernel timings repeat to each context length.
It is the opening 1,277,657 bytes of William H. Prescott's *History of the Conquest of Peru*, from the Project Gutenberg text of ebook 1209 as it was before Gutenberg reformatted it, and it ends mid-sentence.
The book is in the public domain, and the notice is in `THIRD_PARTY_NOTICES.md` at the root of the repository.

The kernels configs read it as `${root:data/prompts/gutenberg_300k.txt}`, so copy it under the shared root once and check the copy against its pin.

```bash
install -D -m 644 data/prompts/gutenberg_300k.txt data/prompts/gutenberg_300k.txt.sha256 -t $LSA_ROOT/data/prompts
(cd $LSA_ROOT/data/prompts && sha256sum -c gutenberg_300k.txt.sha256)
```

The pin is the sha256 `70a4cd68079c541c4bfe6ce19fb0dc4c17774a7885e938fb2b5b63bf856e5988`, and a test holds the file in this directory to it.
Gutenberg's current ebook 1209 is wrapped and headed differently, so downloading it does not give these bytes.
