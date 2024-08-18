---
layout:
  title:
    visible: true
  description:
    visible: true
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
---

# 🤔 Code snippets to get you started



#### BERT

<details>

<summary>BERT "starting" code</summary>

#### Step 1: Install the Required Libraries

If you haven't installed the `transformers` library yet, you can do so using pip:

```bash
pip install transformers
pip install torch
```

#### Step 2: Load a Pre-trained BERT Model and Tokenizer

We'll use the pre-trained `bert-base-uncased` model for this example.

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained BERT model for sequence classification (e.g., sentiment analysis)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Set the model to evaluation mode (this disables dropout layers)
model.eval()
```

#### Step 3: Preprocess Input Text

Next, let's prepare a sample text input and tokenize it using the BERT tokenizer.

```python
# Sample text input
text = "I love using BERT for natural language processing tasks!"

# Tokenize the input text and convert it to PyTorch tensors
inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

# The tokenized inputs include input_ids, attention_mask, etc.
print(inputs)
```

#### Step 4: Perform Inference

Now that we have tokenized the input, we can pass it through the BERT model to get the output. For sequence classification tasks, the output will typically include logits for each class.

```python
# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Get the logits (unnormalized predictions)
logits = outputs.logits

# Convert logits to probabilities
probs = torch.softmax(logits, dim=1)

# Get the predicted class (highest probability)
predicted_class = torch.argmax(probs, dim=1)

# Print the results
print(f"Predicted class: {predicted_class.item()}, Probabilities: {probs}")
```

#### Step 5: Interpreting the Results

In a real-world scenario, you would have multiple classes, such as positive and negative sentiments for sentiment analysis. The `predicted_class` variable will give you the index of the class with the highest probability, which you can then map to your actual class labels.

#### Customization for Fine-Tuning

If you want to fine-tune BERT on a specific dataset, you would typically need to:

1. Load your dataset.
2. Tokenize the text data.
3. Fine-tune the model using your labeled data (requires training).

For simple tasks or experimentation, though, the above example shows how to use a pre-trained BERT model for classification without fine-tuning.

#### Additional Considerations

* **Fine-Tuning:** If you plan to fine-tune BERT for a specific task, you would need to add a training loop where you backpropagate the loss and update the model’s weights using an optimizer.

</details>

{% embed url="https://huggingface.co/docs/transformers/en/model_doc/bert" %}

***

#### GPT-2

Starter code

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Encode input prompt
input_text = "Once upon a time,"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text continuation
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# Decode and print the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```



{% embed url="https://huggingface.co/openai-community/gpt2" %}

***

#### T5

<details>

<summary>Model summary</summary>

T5, or **Text-To-Text Transfer Transformer**, is a versatile and powerful language model developed by Google Research, which is part of their larger exploration into the capabilities of transformers in Natural Language Processing (NLP). T5 stands out for its unique approach to NLP tasks, where it frames all tasks as a text-to-text problem. This means that both the input and output are always text strings, regardless of the specific task, making the model incredibly flexible.

#### **Text-to-Text Framework**

* **Unified Approach:** Unlike models that are specialized for specific tasks (e.g., BERT for classification and GPT-2 for text generation), T5 treats every problem as a text generation problem. For example:
  * **Text Classification:** Input might be "classify: This movie was amazing" with an output of "positive."
  * **Translation:** Input could be "translate English to French: The book is on the table" with an output of "Le livre est sur la table."
  * **Summarization:** Input might be "summarize: The quick brown fox jumps over the lazy dog." with an output of a shortened version.
* **Task Prefixes:** T5 uses prefixes to specify the task at hand, such as "translate English to French:" or "summarize:", which tells the model what type of output is expected.

#### **Architecture**

* **Transformer-Based:** T5 is built on the standard Transformer architecture, utilizing both the encoder and decoder stacks. This is different from models like BERT (which only uses the encoder) or GPT-2 (which only uses the decoder).
* **Scalability:** T5 comes in various sizes, ranging from small versions with 60 million parameters to the large version with 11 billion parameters. The larger the model, the better its performance, albeit at the cost of increased computational resources.

#### **Pretraining**

* **Massive Pretraining Corpus:** T5 was pretrained on the C4 dataset (Colossal Clean Crawled Corpus), which contains hundreds of gigabytes of cleaned text data from the web. This extensive pretraining allows T5 to learn a wide range of language patterns and facts.
* **Self-Supervised Learning:** The model was trained using a self-supervised objective called "span corruption," where a portion of the input text is masked out, and T5 is tasked with generating the missing spans of text. This helps the model learn to generate text that fits naturally within the context of the input.

#### **Fine-Tuning**

* **Task-Specific Fine-Tuning:** After pretraining, T5 can be fine-tuned on specific tasks by providing labeled datasets where both input and output are text. The model then learns to perform the task in a more specialized manner.
* **Flexibility:** The text-to-text framework allows T5 to be fine-tuned for a wide variety of tasks, including translation, summarization, question answering, and more, without changing the model architecture.

#### **Applications**

* **Text Summarization:** T5 can condense long pieces of text into shorter summaries while retaining the essential information.
* **Machine Translation:** T5 can translate text between different languages, making it useful for multilingual applications.
* **Text Classification:** Although traditionally done using models like BERT, T5 can also perform text classification by outputting the appropriate label as text.
* **Question Answering:** T5 can be used to generate answers to questions based on a provided context.
* **Creative Writing:** Like GPT-2, T5 can generate text creatively based on a given prompt, though it is typically used in more structured tasks.

#### **Strengths**

* **Unified Framework:** The ability to handle multiple NLP tasks within a single model architecture is one of T5's greatest strengths. This reduces the need for task-specific models and allows for more straightforward transfer learning.
* **Performance:** T5 has achieved state-of-the-art results on several benchmarks, such as the GLUE benchmark for language understanding, showcasing its strong generalization abilities.
* **Versatility:** T5’s text-to-text paradigm is flexible, making it applicable to almost any text-related task.

#### **Limitations**

* **Resource-Intensive:** Larger versions of T5 require significant computational resources for both training and inference. This makes it less accessible for those without access to high-end hardware.
* **Complexity:** The model’s general-purpose nature means that it might not always be the most efficient choice for highly specialized tasks where simpler models might suffice.
* **Interpretability:** Like many large language models, T5 can be difficult to interpret, making it challenging to understand why it generates specific outputs.

#### Summary

* **T5** is an extremely versatile and powerful model that can handle a wide range of NLP tasks by framing them all as text-to-text problems. Its unified framework makes it a valuable tool for researchers and developers who need a single model capable of performing multiple tasks.
* T5 is especially useful for tasks like summarization, translation, and question answering, where generating high-quality text is essential.t.

</details>

Starter code

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Define the input text and the task (e.g., summarization)
input_text = "summarize: The quick brown fox jumps over the lazy dog. The dog barked in return. They continued to play together in the yard."
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate the output (summary)
outputs = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)

# Decode and print the output
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
In 
```



{% embed url="https://huggingface.co/docs/transformers/en/model_doc/t5" %}

#### Bart

<details>

<summary>Model summary</summary>

BART, which stands for **Bidirectional and Auto-Regressive Transformers**, is a powerful sequence-to-sequence model developed by Facebook AI (now Meta AI). BART is designed to handle a variety of Natural Language Processing (NLP) tasks, particularly those that involve generating text, such as text summarization, translation, and text generation. It combines ideas from both BERT (Bidirectional Encoder Representations from Transformers) and GPT (Generative Pre-trained Transformer) to create a model that can understand and generate text effectively.

#### **Architecture**

* **Encoder-Decoder Model:** BART is a sequence-to-sequence model, meaning it has an encoder-decoder structure. The encoder processes the input text, and the decoder generates the output text. This architecture is similar to models like T5, which also uses an encoder-decoder structure.
* **Combination of BERT and GPT:** BART can be seen as a hybrid between BERT and GPT:
  * **BERT-like Encoder:** The encoder is similar to BERT and reads the input text in a bidirectional manner, meaning it considers the context from both the left and the right of each word.
  * **GPT-like Decoder:** The decoder is similar to GPT and generates text in an auto-regressive manner, meaning it predicts each word in a sequence based on the previous words.

#### **Training Objective**

* **Denoising Autoencoder:** BART is trained as a denoising autoencoder. During training, the model is fed corrupted versions of text, where some parts of the text have been randomly masked, deleted, or rearranged. The model’s task is to reconstruct the original, uncorrupted text. This training strategy helps BART learn to generate coherent and contextually appropriate text, even when the input is noisy or incomplete.
* **Corruption Techniques:** The training process uses various corruption techniques, such as:
  * **Token Masking:** Similar to BERT, some tokens in the input text are masked, and the model must predict them.
  * **Text Infilling:** Random spans of text are replaced with a mask token, and the model must fill in the missing spans.
  * **Sentence Shuffling:** The order of sentences in the text is shuffled, and the model must reorder them correctly.
  * **Document Rotation:** The text is rotated by a random number of tokens, and the model must unrotate it.

#### **Applications**

* **Text Summarization:** BART is particularly strong at summarizing long pieces of text into concise summaries. It has been fine-tuned on summarization datasets like CNN/DailyMail and has achieved state-of-the-art results in this task.
* **Machine Translation:** BART can be fine-tuned to translate text from one language to another, similar to models like T5.
* **Text Generation:** The model can generate text based on a given prompt, making it useful for tasks like story generation, dialogue generation, or any other creative writing tasks.
* **Text Restoration:** Given its training objective, BART is well-suited to tasks that involve restoring or correcting text, such as error correction or text completion.

#### **Strengths**

* **Versatility:** BART’s combination of a bidirectional encoder and an autoregressive decoder makes it highly versatile and effective for a range of text generation tasks.
* **State-of-the-Art Performance:** BART has achieved state-of-the-art results on several benchmarks, particularly in text summarization. Its ability to handle noisy inputs and generate coherent outputs is one of its key strengths.
* **Flexibility in Fine-Tuning:** Like T5, BART can be fine-tuned for specific tasks, allowing it to adapt to a wide variety of applications, from summarization to translation to dialogue generation.

#### **Limitations**

* **Resource-Intensive:** BART’s encoder-decoder architecture, especially in larger versions, can be resource-intensive to train and deploy, requiring significant computational resources.
* **Complexity:** The training process involving multiple corruption strategies adds complexity to the model, making it less straightforward to understand and implement compared to simpler models like GPT-2.

#### **Comparison with Other Models**

* **BERT vs. BART:**
  * BERT is primarily designed for understanding tasks (e.g., classification, NER) and is not inherently a generative model. BART, on the other hand, is designed for both understanding and generation tasks, making it more versatile.
* **GPT-2 vs. BART:**
  * GPT-2 is excellent for generating text in an autoregressive manner but lacks the bidirectional understanding that BART’s encoder provides. BART’s encoder-decoder structure allows it to both understand and generate text, giving it an edge in tasks like summarization and translation.
* **T5 vs. BART:**
  * Both T5 and BART are sequence-to-sequence models, but BART’s training as a denoising autoencoder with various corruption techniques makes it particularly robust to noisy inputs. T5 is also highly versatile but might require more careful task framing (using specific task prefixes).

#### Summary

* **BART** is a versatile and powerful sequence-to-sequence model that excels in text generation tasks, particularly in scenarios where the input may be noisy or incomplete. Its encoder-decoder architecture allows it to perform a wide range of tasks, from summarization and translation to text generation and restoration.
* **Strengths:** BART’s robust performance in handling noisy inputs and generating coherent text, combined with its versatility across tasks, makes it a valuable tool in NLP.
* **Applications:** BART is particularly well-suited for summarization, translation, and other text generation tasks, where both understanding and generating text are crucial.



</details>

Starter code

```python
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the pre-trained BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Define the input text
text = """The quick brown fox jumps over the lazy dog. The dog barked in return. They continued to play together in the yard, enjoying the sunny day."""

# Encode the input text and generate the summary
inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

# Decode and print the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
```

{% embed url="https://huggingface.co/docs/transformers/en/model_doc/bart" %}
