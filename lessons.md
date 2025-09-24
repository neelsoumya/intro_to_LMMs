

# Lesson Series: Introduction to Large Language Models (LLMs)

Below are ready-to-use **Markdown** lesson pages for Lessons 1–8. Copy each lesson into your course site, notebook descriptions, or slide notes.

---

## Lesson 1 — What are LLMs? Big picture & intuition

### Objectives

* Understand what LLMs are, why they work at a high level, and common applications.  
* Develop an intuitive visual model of how LLMs process language.

### Required viewing / reading

* FT visual explainer (very accessible, visual): [https://ig.ft.com/generative-ai/](https://ig.ft.com/generative-ai/)  
* Course intro notes (theory primer): [https://docs.science.ai.cam.ac.uk/large-language-models/Introduction/Introduction/](https://docs.science.ai.cam.ac.uk/large-language-models/Introduction/Introduction/)

### Lecture outline

1. Course intro & learning outcomes  
2. High-level history: n-grams → RNNs → attention → transformers → LLMs  
3. Demonstration of an LLM (live demo or simple notebook)

### In-class activity / lab

* Walk through the FT visual explainer together; short quiz / discussion.  
* Short Colab notebook: query a small open-source model (or hosted demo) and inspect token outputs.

[Notebook exercise](https://github.com/neelsoumya/intro_to_LMMs/blob/main/LLM_demo_tokenlevel_inspection.ipynb)

### Assignment

* Write a 300–500 word reflection: *How does the model produce text?* Cite the FT explainer and the course intro notes.

---

## Lesson 2 — Next-token prediction & training objective

### Objectives

* Understand next-token prediction formally and practically.  
* Connect loss functions (cross-entropy) to the training task.

### Required reading

* Next token prediction primer: [https://medium.com/@akash.kesrwani99/understanding-next-token-prediction-concept-to-code-1st-part-7054dabda347](https://medium.com/@akash.kesrwani99/understanding-next-token-prediction-concept-to-code-1st-part-7054dabda347)

### Lecture outline

* Formal definition of next-token prediction  
* Sequence modeling vs conditional generation  
* Loss and evaluation metrics (perplexity, token accuracy)

### Lab

* Small coding exercise: implement a toy next-token predictor (character or word level) in Python (pure NumPy or PyTorch skeleton).

### Assignment

* Short notebook: train a toy predictor on a tiny dataset and report perplexity; include 1–2 short examples of generated continuations.

---

## Lesson 3 — Mathematical foundations & intuition

### Objectives

* Strengthen the mathematical background needed (vectors, matrices, projections, softmax).  
* Connect intuitive explanations to linear algebra operations used in LLMs.

### Required viewing / reading

* 3Blue1Brown deep learning playlist (select videos): [https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)  
* Mathematical foundations playlist / repo: [https://www.youtube.com/watch?v=nfZQYopzv20&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=3&t=173s](https://www.youtube.com/watch?v=nfZQYopzv20&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu&index=3&t=173s) and [https://github.com/Chandan-IISc/IITM_GenAI](https://github.com/Chandan-IISc/IITM_GenAI)

### Lecture outline

* Vector spaces, dot products, projections, softmax and logit intuition  
* Why embeddings are vector representations of meaning

### Lab

* Visualize embeddings for a small vocabulary using PCA / t-SNE (simple notebook).  
* Short derivation/exercise on softmax gradients.

### Assignment

* Report with plots and brief interpretation of embedding geometry.

---

## Lesson 4 — Attention mechanism (deep dive)

### Objectives

* Understand scaled dot-product attention, Q/K/V decomposition and multi-head attention.  
* Explore how attention enables context-sensitive representations.

### Required viewing / reading

* Intro to attention (video): [https://www.youtube.com/watch?v=XN7sevVxyUM](https://www.youtube.com/watch?v=XN7sevVxyUM)  
* Illustrated transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)  
* Wikipedia attention (reference figures): [https://en.wikipedia.org/wiki/Attention_(machine_learning)](https://en.wikipedia.org/wiki/Attention_%28machine_learning%29)

### Lecture outline

* Q/K/V intuition and formulas; scaled dot product; masking and causal attention  
* Multi-head attention: why multiple heads help

### Lab

* Implement attention from scratch (NumPy or PyTorch) and visualize attention weights on a short sentence.  
* Compare single-head vs multi-head attention qualitatively.

### Assignment

* Notebook implementing attention + one page write-up interpreting attention maps.

---

## Lesson 5 — Transformers (architecture & training at scale)

### Objectives

* Combine attention with feedforward and positional encodings to form the transformer block.  
* Understand encoder vs decoder vs decoder-only (GPT family).

### Required viewing / reading

* 3Blue1Brown transformer video: [https://www.youtube.com/watch?v=eMlx5fFNoYc&vl=en](https://www.youtube.com/watch?v=eMlx5fFNoYc&vl=en)  
* Karpathy — Build GPT-2 from first principles: [https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)  
* Cheatsheet: [https://github.com/afshinea/stanford-cme-295-transformers-large-language-models/blob/main/en/cheatsheet-transformers-large-language-models.pdf](https://github.com/afshinea/stanford-cme-295-transformers-large-language-models/blob/main/en/cheatsheet-transformers-large-language-models.pdf)

### Lecture outline

* Transformer block components; residuals and layernorm  
* Positional encodings and sequence processing  
* Model sizes and scaling properties

### Lab

* Read Karpathy video companion code / implement a tiny transformer and run a short training loop.  
* Use the cheatsheet to map formulas to code implementations.

### Assignment

* Short report: describe the forward pass of a transformer block with annotated code snippets.

---

## Lesson 6 — Embeddings, similarity and retrieval-augmented generation (RAG)

### Objectives

* Use embeddings for semantic search and retrieval augmentation.  
* Build a simple retrieval-augmented generation (RAG) pipeline.

### Required resources

* Embedding video (3Blue1Brown content): [https://www.youtube.com/watch?v=wjZofJX0v4M](https://www.youtube.com/watch?v=wjZofJX0v4M)  
* LangChain + Hugging Face integration: [https://python.langchain.com/docs/integrations/chat/huggingface/](https://python.langchain.com/docs/integrations/chat/huggingface/)

### Lecture outline


## Quick links

* FT visual explainer: [https://ig.ft.com/generative-ai/](https://ig.ft.com/generative-ai/)
* Course intro notes (theory primer): [https://docs.science.ai.cam.ac.uk/large-language-models/Introduction/Introduction/](https://docs.science.ai.cam.ac.uk/large-language-models/Introduction/Introduction/)
* Karpathy — Build GPT-2 from first principles: [https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)
* Vizuara playlist & newsletter: [https://www.youtube.com/watch?v=Xpr8D6LeAtw\&list=PLPTV0NXA\_ZSgsLAr8YCgCwhPIJNNtexWu](https://www.youtube.com/watch?v=Xpr8D6LeAtw&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu)
  & [https://www.vizuaranewsletter.com/p/9e1](https://www.vizuaranewsletter.com/p/9e1)
* 3Blue1Brown playlist & resources: [https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1\_67000Dx\_ZCJB-3pi](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
  & [https://www.3blue1brown.com/lessons/gpt](https://www.3blue1brown.com/lessons/gpt)
* Illustrated transformer (Jalammar): [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
* LangChain + Hugging Face integration: [https://python.langchain.com/docs/integrations/chat/huggingface/](https://python.langchain.com/docs/integrations/chat/huggingface/)
* Context engineering tutorial: [https://towardsdatascience.com/context-engineering-a-comprehensive-hands-on-tutorial-with-dspy/](https://towardsdatascience.com/context-engineering-a-comprehensive-hands-on-tutorial-with-dspy/)
* Mathematical foundations repo: [https://github.com/Chandan-IISc/IITM\_GenAI](https://github.com/Chandan-IISc/IITM_GenAI)
* Cheatsheet: [https://github.com/afshinea/stanford-cme-295-transformers-large-language-models/blob/main/en/cheatsheet-transformers-large-language-models.pdf](https://github.com/afshinea/stanford-cme-295-transformers-large-language-models/blob/main/en/cheatsheet-transformers-large-language-models.pdf)
* Awesome generative AI guide: [https://github.com/aishwaryanr/awesome-generative-ai-guide](https://github.com/aishwaryanr/awesome-generative-ai-guide)

---

## Course summary

**Duration (suggested):** 8 lessons (each lesson ≈ 2–3 hours: 60–90 min lecture + 60–90 min lab)

**Goals:**

* Provide an intuitive and practical introduction to LLMs and transformers.
* Teach foundational mathematics and implementation details (attention, transformer blocks).
* Provide hands-on experience with open-source tooling (Hugging Face, LangChain).
* Cover context/prompt engineering, retrieval augmentation (RAG), and ethical considerations.

**Target audience:** Advanced undergraduates, master’s students, and practitioners with basic ML familiarity.

**Assessment (suggested):**

* Final project (70%): working code + report (≤10 pages) + demo (10 min).
* Assignments & labs (20%): cumulative notebooks and short writeups.
* Participation & quizzes (10%).

---

## Learning outcomes

By the end of the course students will be able to:

* Explain the architecture and training objective of LLMs (next-token prediction).
* Derive and implement attention and understand transformers at a code level.
* Use embeddings and retrieval to augment context for generation.
* Build small applications with LangChain and Hugging Face.
* Apply prompt/context engineering techniques and critique model outputs.
* Describe limitations, biases, and ethical implications of deployed LLMs.

---

## Lesson plan (files & notebooks)

Each lesson should have a short slide deck, a runnable notebook, and an assignment sheet.

**Notebooks (suggested):**

* `notebooks/01_intro.ipynb` — What are LLMs? Big picture + demo
* `notebooks/02_next_token.ipynb` — Next-token prediction + toy implementation
* `notebooks/03_math_foundations.ipynb` — Vectors, softmax, projections, embeddings
* `notebooks/04_attention_implementation.ipynb` — Implement scaled dot-product attention
* `notebooks/05_tiny_transformer.ipynb` — Tiny transformer / training loop
* `notebooks/06_embeddings_rag.ipynb` — Embeddings, semantic search, simple RAG pipeline
* `notebooks/07_prompt_engineering.ipynb` — Prompt variants, few-shot examples, evaluation
* `notebooks/08_tools_ethics.ipynb` — Tooling walkthrough + ethics exercises

**Slides:**

* `slides/lesson01_intro.pdf` … `slides/lesson08_tools_ethics.pdf`

**Assignments & rubrics:**

* `assignments/` — assignment specs, datasets, submission templates, rubrics

**Final project template:**

* `final_project/PROPOSAL_TEMPLATE.md`
* `final_project/RUBRIC.md`
* `final_project/DEMO_CHECKLIST.md`

**Other:**

* `REFERENCES.md` — curated list of links and brief notes (see Quick links above)
* `logo.png` — repository logo used in README and slides

---

## Detailed lesson mapping (short)

* Lesson 1 — *What are LLMs?* — FT explainer + course intro notes; in-class demo
* Lesson 2 — *Next-token prediction* — loss functions, toy trainer
* Lesson 3 — *Math foundations* — softmax, dot products, embedding geometry
* Lesson 4 — *Attention deep dive* — Q/K/V, masking, multi-head attention
* Lesson 5 — *Transformers* — block composition, positional encodings, Karpathy walkthrough
* Lesson 6 — *Embeddings & RAG* — cosine similarity, indexes, LangChain + HF example
* Lesson 7 — *Context engineering* — prompt patterns, few-shot, evaluation
* Lesson 8 — *Tools, ethics & project kickoff* — Hugging Face, LangChain, risks and deployment

---

## Suggested schedule (compact)

* Week 1: Lessons 1 & 2
* Week 2: Lesson 3
* Week 3: Lesson 4
* Week 4: Lesson 5
* Week 5: Lesson 6
* Week 6: Lesson 7
* Week 7: Lesson 8 + project kickoff
* Weeks 8–10: Project work, demos, and final presentations

---

## How to run the notebooks

1. Prefer Colab for lower friction: add a link header in each notebook: "Open in Colab".
2. Use lightweight models or small datasets so notebooks run on CPU or a modest GPU.
3. Provide requirements files for reproducibility: `requirements.txt` (recommended minimal pins). Example:

```
# example requirements.txt
numpy
scipy
scikit-learn
torch
transformers
datasets
langchain
faiss-cpu  # or faiss-gpu if available
jupyter
matplotlib
pandas
```

4. Include a `setup.md` for environment setup and a `data/` folder or small seed data for exercises.

---

## Teaching tips for instructors

* Ask students to skim the FT explainer before Lesson 1.
* Keep notebooks minimal and reproducible; prefer small models and short training runs.
* Use Colab links to reduce environment setup friction.
* Make rubrics explicit and include reproducibility checks.
* Reserve a full session for ethics and responsible deployment.

---

## Final project ideas

* RAG-based QA assistant over a curated document set.
* Attention visualiser and interpretability study with short report.
* Prompt-engineering toolkit for a specific task with robustness evaluation.

---

## Contributing

Contributions are welcome! Please open issues for:

* Bug reports in notebooks
* Suggestions for lecture content or additional resources
* New labs, datasets, or project ideas

Recommended workflow:

1. Fork the repository
2. Create a topic branch
3. Submit a pull request with a clear description and any required dataset links

---

## License & citation

Add a LICENSE file to the repository (e.g., MIT or CC-BY) and a short citation guide for students who reuse course materials.

---

## Contact / Maintainer

Soumya Banerjee

