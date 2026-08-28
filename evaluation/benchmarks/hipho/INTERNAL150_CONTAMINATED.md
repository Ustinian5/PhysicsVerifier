This directory's previous `hipho_answer_acc` numbers are an internal contaminated diagnostic.

They were computed on a 150-row expansion file that is **not** official SciYu/HiPhO, using binary boxed-answer matching rather than the paper's answer-level + marking-scheme step-level + exam/MNS protocol. The file also leaked into `swift_prompts` (exact IDs and normalized question hashes).

Do not overwrite these files. Do not use them for paper comparison or for claiming training onset.
