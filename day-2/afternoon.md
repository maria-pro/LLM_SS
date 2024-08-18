---
description: Hands-On with LLMs in Social Science Research
---

# ⚙️ Afternoon

#### <mark style="color:purple;">Slides</mark>



{% file src="../.gitbook/assets/2_2. More LLM.pptx" %}

***



### Tokenizers

* **Tokenizers** in Hugging Face are essential tools for converting text into a format that models can understand, processing the text into tokens, adding special tokens, padding/truncating sequences, and converting tokens back into text.
* **Different tokenization methods** (WordPiece, BPE, Unigram, SentencePiece) offer flexibility in handling various languages and tasks, depending on the needs of the model.



### Key Hugging Face models:

#### BERT&#x20;

<details>

<summary>Model summary</summary>

#### **Architecture**

* **Transformer Model:** BERT is based on the Transformer architecture, which relies on self-attention mechanisms to process words in a sentence relative to each other, regardless of their position. This is a significant departure from previous models that processed text sequentially.
* **Bidirectional:** Unlike previous models like GPT (which is unidirectional), BERT reads text bidirectionally, meaning it considers the context from both the left and the right of a given word. This allows BERT to better understand the meaning of words in context, which is crucial for many NLP tasks.

#### **Pretraining Tasks**

BERT was pretrained on two specific tasks that enable it to develop a deep understanding of language:

* **Masked Language Modeling (MLM):** During pretraining, some percentage of the input tokens are randomly masked, and BERT's goal is to predict these masked tokens. This forces the model to learn how words relate to each other in context.
* **Next Sentence Prediction (NSP):** BERT is trained to understand the relationship between two sentences. It is given pairs of sentences and must predict whether the second sentence follows the first one in the original text. This helps BERT understand sentence-level relationships.

#### **Fine-Tuning**

* After pretraining, BERT can be fine-tuned on specific tasks with relatively small amounts of data. Fine-tuning involves slightly modifying the pretrained BERT model on a specific dataset, such as for sentiment analysis, named entity recognition, or question answering.
* BERT has been shown to achieve state-of-the-art results on many NLP tasks by simply adding a task-specific output layer and fine-tuning the model on task-specific data.

</details>

{% embed url="https://huggingface.co/docs/transformers/en/model_doc/bert" %}



#### GPT-2



<details>

<summary>Model summary</summary>

#### **Architecture**

* **Transformer Model:** GPT-2 is based on the Transformer architecture, specifically using the decoder portion of the Transformer. This architecture uses self-attention mechanisms to process and generate text, allowing the model to consider the context provided by the input text.
* **Unidirectional:** Unlike BERT, which is bidirectional, GPT-2 processes text in a unidirectional manner, meaning it predicts the next word in a sequence based solely on the preceding words. This makes it particularly effective at generating coherent, contextually relevant text.

#### **Pretraining**

* **Large-scale Pretraining:** GPT-2 was pretrained on a massive dataset of 8 million web pages, giving it a broad understanding of language, facts, and context. The model learns to predict the next word in a sentence, which helps it generate fluent and contextually appropriate text.
* **Zero-shot Learning:** During pretraining, GPT-2 was not specifically trained for particular tasks like translation or summarization. However, it can perform these tasks without additional training (zero-shot), based on its understanding of language from pretraining.

#### **Model Variants**

* GPT-2 was released in multiple sizes, each with a different number of parameters:
  * **Small:** 117 million parameters.
  * **Medium:** 345 million parameters.
  * **Large:** 762 million parameters.
  * **Extra Large:** 1.5 billion parameters.
* The larger the model, the better it typically performs at generating coherent and contextually relevant text, though it also requires more computational resources.

#### **Text Generation Capabilities**

* **Coherent Text Generation:** GPT-2 is particularly known for its ability to generate coherent and fluent text that often mimics human writing. It can continue a given text prompt, generate creative writing, answer questions, or even create dialogue.
* **Context Sensitivity:** The model generates text that aligns with the context provided by the input prompt, making it useful for various applications where context-aware text is required.

#### **Applications**

* **Creative Writing:** GPT-2 can generate stories, poems, dialogues, or any other form of creative writing, making it a tool for writers looking for inspiration or assistance in drafting content.
* **Chatbots:** GPT-2 can be used to build conversational agents that engage users in natural and coherent conversations.
* **Content Creation:** It can help in creating content for blogs, articles, or social media posts by generating drafts or even full pieces based on a brief prompt.
* **Code Generation:** GPT-2 can also be adapted to generate code snippets or provide suggestions in programming, although GPT-3 and later models are more commonly used for this purpose.
* **Language Translation:** While not specifically trained for translation, GPT-2 can generate translations by providing it with a prompt that suggests a translation task.
* **Summarization:** By providing a prompt that asks for a summary, GPT-2 can generate summaries of longer texts, though models like T5 and BART are often better suited for this task.

#### **Strengths**

* **Versatility:** GPT-2’s ability to handle a wide range of tasks without task-specific training makes it extremely versatile.
* **Fluency:** The model generates text that is fluent and contextually appropriate, often indistinguishable from text written by humans.

#### **Limitations**

* **Lack of Specificity:** Because GPT-2 wasn’t trained on specific tasks, it may not always produce accurate or task-optimized outputs (e.g., translations or summaries might not be as precise as those from models specifically trained for these tasks).
* **Bias:** Like many large-scale language models, GPT-2 can reflect biases present in the training data, potentially generating biased or inappropriate content.
* **Compute Intensive:** The larger versions of GPT-2 require substantial computational resources for both training and inference.

#### **Impact and Reception**

* GPT-2’s release marked a significant milestone in NLP, particularly for its ability to generate coherent text at scale. Its capabilities demonstrated the potential of large language models for a wide range of applications.
* Initially, OpenAI was cautious about releasing the full model due to concerns about its potential misuse, such as generating fake news or spam. This led to a broader conversation about the ethical implications of powerful language models.

</details>
