# [COURSE] LLMs From Scratch

### Course Overview
Are you relatively technical, interested in Large Language Models, but have yet to dive into the interworkings of these models? Same for us - That's why we created this course.

Topics Covered:
- Tokenization
- Embeddings
- Transformers / Attention Mechanisms
- Pre-Training LLMs
- Fine-tuning LLMs

### Objectives
The goal of this course is that each participant leaves with a granular understanding and ability to manipulate LLMs from architecture design to fine-tuning. We view this as a foundational building block for further research into advanced topics and/or building applications with LLMs.

### Format
- There is no teacher for this course, rather this is a group study of carefully selected materials. 
- We will meet once a month in a casual interactive setting to discuss the materials, answer questions, and work on exercises together. We will also use Discourse for inter-meeting discussions.

### Book [required]
We will be working through a number of resources, most of which are freely available online. However we will be relying heavily on the following required text: 
- "Build a Large Language Model (From Scratch)" by Sebastian Raschka
- Available as PDF, EPUB https://www.manning.com/books/build-a-large-language-model-from-scratch
- Discount code: SRLLM24
- Full code repo: https://github.com/rasbt/LLMs-from-scratch

### Activites
- Reading: Read = read, run the code, and attempt the exercises
- Co-work: deliberate practice, debug, try to break things
- Group: Monthly meetings to summarize key insights, review code together, Q&A (each meeting someone will volunteer to present)
- Discord: chat about any questions/issues/discoveries

### Projects
- Build a working BPE tokenizer from scratch
- Experiment with your own custom transformer block with multi-head self attention
- Pre-train and fine-tune a 1.5B parameter LLM on a publicly available dataset
- Build GPT-2 entirely from scratch




## Proposed Schedule [*Needs to be updated from Notion*]

Proposed Dates

- Meeting 0 [Intro] - Monday Oct 28
- Meeting 1 [Tok/Emb] - Monday Nov 18
- Meeting 2 [Attention] - Monday Dec 16
- Meeting 3 [LLM Arch] - Monday Jan 20
- Meeting 4 [Pre-training] - Monday Feb 20
- Meeting 5 [Fine-tuning] - Monday Mar 17
- Meeting 6 [Project Showcase] - Wed Apr 2


### Meeting 0 (Oct 2024): Introduction and Setup
Materials

- [Beginners] Intro to [LLMs Karpathy video](https://www.youtube.com/watch?v=zjkBMFhNj_g&t=1s&ab_channel=AndrejKarpathy)
- [Beginners] Python [online course](https://programming-24.mooc.fi/part-1/1-getting-started)
- [Beginners] [Kaggle ML learning modules](https://www.kaggle.com/learn) (first 6 modules)
- [Beginners] Pytorch step-by-step Tutorial (add link to notebook)
- [Beginners] Appendix A: Intro to Pytorch (not enough code)
- [All] Google [crash course: What are LLMs](https://developers.google.com/machine-learning/crash-course/llm) → align terminologies
- [All] List of key concepts (add link) → align terminologies
- [All] Read* Chapters 1 of the Book
- [All]
- [All] LLM-Workshop-2024 Video and Repo from SR https://github.com/rasbt/LLM-workshop-2024 (video to be released from Scipy conference)

Estimate workload:

- [Beginners] about 20h [Python], 10h [ML] , 10h [PyTorch]
- [All] 5-10h

Group: Overview plan, setup workflow, align terminologies, Python & Pytorch warmup (train a NN or do a Kaggle challenge for swag)


### Month 2: Nov 2024

Materials

- Chapter 2 Working with Text Data : read and code
- From NLP to token prediction - a brief history (add link)
- Watch [Karpathy's video on training a BPE](https://youtube.com/watch?v=zduSFxRajkE) tokenizer from scratch - Do the Colab notebook.

Estimate workload: 10-20h


### Month 3 : Dec 2024

Materials

- Chapter 3 on Attention mechanism: read and code
- Watch [Stanford CS25: Karpathy on Transformers](https://www.youtube.com/watch?v=XfpMkf4rD6E&ab_channel=StanfordOnline)
- Maths behind Attention mechanism explained step by step (add link)

Question: is it ok to separate chapter 3 and 4? I feel like a lot can be said for Attention

Look for some more resources

Group: TBD

### Mid-term Break: Build a project or demo
Add 1 month to build something

### Month 4 : Feb 2025

Materials: 

- Chapter 4 on implementing the model architecture.
- Watch Karpathy's video on pretraining the LLM.

Group: TBD


### Month 5 : Mar 2025

Materials

- Read Chapter 5 on pre-training the LLM and then loading pretrained weights.
- Read Appendix E on adding additional bells and whistles to the training loop.


### Month 6 : Apr 2025

Reading:

- Read Chapters 6 and 7 on fine-tuning the LLM.
- Read Appendix E on parameter-efficient fine-tuning with LoRA.

Group: TBD
