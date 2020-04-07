# Reading List
---

## NLP Papers

- [ ] [A Primer in BERTology: What we know about how BERT works](https://arxiv.org/pdf/2002.12327.pdf)
- [ ] [MASKGAN: BETTER TEXT GENERATION VIA FILLING IN THE ___ ](https://arxiv.org/pdf/1801.07736.pdf)
- [ ] [COMPRESSIVE TRANSFORMERS FOR LONG-RANGE SEQUENCE MODELLING](https://arxiv.org/pdf/1911.05507.pdf)
- [ ] [What Does BERT Look At? An Analysis of BERT's Attention](https://arxiv.org/abs/1906.04341)
- [ ] [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860.pdf)
- [ ] [Visualizing and Understanding Neural Models in NLP](https://www.aclweb.org/anthology/N16-1082.pdf)
- [ ] [Label-Agnostic Sequence Labeling by Copying Nearest Neighbors](https://arxiv.org/pdf/1906.04225.pdf)
- [ ] [GENERATING WIKIPEDIA BY SUMMARIZING LONG SEQUENCES](https://arxiv.org/pdf/1801.10198.pdf)
- [ ] [Fixed Encoder Self-Attention Patterns in Transformer-Based Machine Translation](https://arxiv.org/pdf/2002.10260.pdf) <br />
    simplify encoder self-attention of Transformer-based NMT models by replacing all but one attention head with fixed positional attentive patterns that require neither training nor external knowledge.
    Improve translation quality in low-resource settings thanks to the strong injected prior knowledge about positional attention
- [ ] [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) <br />
    Discuss transformer architecture and first archi to get rid of CNN and RNN (which are computatinally expensive and cannot be parallelised). BERT is based on Transformers

- [ ] [BERT](https://arxiv.org/pdf/1810.04805.pdf) <br />
    Use transfomers to come up with deep bidirectional model to encode both L2R and R2L contexts. Simplify task-specific architecture by re-using pre-training archi also for finetuning method (all parameters are updated end-to-end but quickly). Works well for feature-based approaches as well. Using MLM and NSP objective (based on utility for downstream tasks).

- [ ] [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)<br />
    GPT-1 transformer uses constrained self-attention where every token can only attend to context to its left. This way of using transformer is called "Transformer decoding" since this can be used in text generation (auto-regressive style of decoding).

- [ ] [THIEVES ON SESAME STREET! MODEL EXTRACTION OF BERT-BASED APIS](https://arxiv.org/pdf/1910.12366.pdf)


## CS 533: Natural Language Processing (NLP)

- [ ] [Course Webpage](http://karlstratos.com/teaching/cs533spring20/cs533spring20.html)
- [ ] [Variational and Information Theoretic Principles in Neural Networks](http://karlstratos.com/notes/varinfo.pdf)
- [ ] [Generating Sentences from a Continuous Space](https://arxiv.org/pdf/1511.06349.pdf)
- [ ] [Expectation Maximization (EM)](https://ttic.uchicago.edu/~dmcallester/ttic101-07/lectures/em/em.pdf)
- [ ] [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/pdf/1704.00051.pdf)
- [ ] [Latent Retrieval for Weakly Supervised Open Domain Question Answering](https://www.aclweb.org/anthology/P19-1612.pdf)

## Summarization Papers
- [ ] [On Extractive and Abstractive Neural Document Summarization with Transformer Language Models](https://arxiv.org/pdf/1909.03186.pdf)

- [ ] [Faithful to the Original - Fact Aware Neural Abstractive Summarization](https://arxiv.org/pdf/1711.04434.pdf) <br />
Augment the attention mechanism of neural models with factual triples extracted with open information extraction system <br/>
Ensure the Correctness of the Summary: Incorporate Entailment Knowledge into Abstractive Sentence Summarization
entailment aware encoder (MTL) and entailment aware decoder (Entailment reward maximisation)


- [ ] [Evaluating the Factual Consistency of Abstractive Text Summarization](https://arxiv.org/pdf/1910.12840.pdf)

- [ ] [Assessing The Factual Accuracy of Generated Text](https://arxiv.org/pdf/1905.13322.pdf) <br />
compared different information extraction systems to evaluate the factual accuracy of generated text <br/>

- [ ] [Optimizing the Factual Correctness of a Summary: A Study of Summarizing Radiology Reports](https://arxiv.org/pdf/1911.02541.pdf)

- [ ] [Retrieve, Rerank and Rewrite - Soft Template Based Neural Summarization](https://www.aclweb.org/anthology/P18-1015.pdf)

- [ ] [Ranking Generated Summaries by Correctness](https://www.aclweb.org/anthology/P19-1213.pdf)<br />
 An Interesting but Challenging Application for Natural Language Inference <br/>
Studied whether existing natural language infer- ence systems can be used to evaluate the factual correctness of generated summaries, and found models trained on existing datasets to be inade- quate for this task. <br />

- [ ] [GENERATING WIKIPEDIA BY SUMMARIZING LONG SEQUENCES](https://arxiv.org/pdf/1801.10198.pdf)<br />
Also, we found this very relevant paper that does something similar to our core idea- extract crucial parts from long documents, and then use abstractive ways to summarize. However, they used simple extractive methods for first stage filtering whereas our ideas can be fancier and incorporate more intuition about 'facts'. Also, they have used older SOTA models for step 2 (abstractive summarization) whereas we have better alternatives available now. They also hint at "..results...suggesting future work in improving the extraction step could result in significant improvements. One possibility is to train a supervised model to predict relevance which we leave as future work". Their major contribution is to modify the transformer architecture to introduce a Transformer decoder which supports really long documents, however, they did this for multi-document summarization scenario. They also mention "for our task optimizing for perplexity correlates with increased ROUGE and human judgment. As perplexity decreases we see improvements in the model outputs, in terms of fluency, factual accuracy, and narrative complexity"  so proceeding with the perplexity idea (for extractor) we discussed last time could be good. 

- [ ] [Bottom-Up Abstractive Summarization](https://arxiv.org/pdf/1808.10792.pdf)


## Information Retrieval

- [ ] [An Introduction to Neural Information Retrieval](https://www.microsoft.com/en-us/research/uploads/prod/2017/06/fntir2018-neuralir-mitra.pdf)

- [ ] [A Deep Look into Neural Ranking Models for Information Retrieval](http://www.bigdatalab.ac.cn/~gjf/papers/2019/Survey_Preprint.pdf)

- [ ] [TREC-2019-Deep-Learning](https://microsoft.github.io/TREC-2019-Deep-Learning/)

- [ ] [MSMARCO](http://www.msmarco.org/)

### Talks

- [ ] [Neural IR by Bhaskar Mitra](https://www.youtube.com/watch?v=g1Pgo5yTIKg)

- [ ] [Embeddings for Everything: Search in the Neural Network Era](https://www.youtube.com/watch?v=JGHVJXP9NHw)


## Machine Learning, Deep Learning

- [ ] [YOUR CLASSIFIER IS SECRETLY AN ENERGY BASED MODEL AND YOU SHOULD TREAT IT LIKE ONE](https://arxiv.org/pdf/1912.03263.pdf)

- [ ] [The Consciousness Prior](https://arxiv.org/pdf/1709.08568.pdf)

- [ ] [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf)

- [ ] [Contrastive Self-Supervised Learning](https://ankeshanand.com/blog/2020/01/26/contrative-self-supervised-learning.html)


## Blogs

- [ ] [A quick summary of modern NLP methods](https://medium.com/dair-ai/deep-learning-for-nlp-an-overview-of-recent-trends-d0d8f40a776d)
- [ ] [Complete Modern NLP Survey](https://github.com/omarsar/nlp_overview)
- [ ] [NLP Pretraining](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html)
- [ ] [NLP Applications](https://d2l.ai/chapter_natural-language-processing-applications/index.html)
- [ ] [When Not to Choose the Best NLP Model](https://blog.floydhub.com/when-the-best-nlp-model-is-not-the-best-choice/)

- [ ] [Are Sixteen Heads Really Better than One?](https://blog.ml.cmu.edu/2020/03/20/are-sixteen-heads-really-better-than-one/)

- [ ] [NLP Year in Review - 2019](https://medium.com/dair-ai/nlp-year-in-review-2019-fb8d523bcb19)

- [ ] [Language Models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html)

- [ ] [RNN – Andrej Karpathy’s blog The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

- [ ] [LSTM – Christopher Olah’s blog Understanding LSTM Networks  and R2Rt.com Written Memories: Understanding, Deriving and Extending the LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) <br />
    Use of RNN (Sequential data): when we don’t need any further context – it’s pretty obvious the next word is going to be "sky". In such cases, where the gap between the relevant information and the place that it’s needed is small, RNNs can learn to use the past information. Unfortunately, as that gap grows, RNNs become unable to learn to connect the information.
    “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged. The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.
    helps in gradient flow and structured gates helps adds more flexibility to the model.
    
- [ ] [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [ ] [Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
- [ ] [Attention – Christopher Olah Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/) <br />
    Discusses use of attention for various applications like translation, image captioning and audio transcribing
    
- [ ] [Attention basics](https://mchromiak.github.io/articles/2017/Sep/01/Primer-NN/#attention-basis)
    
- [ ] [Attenion is not not explaination](https://medium.com/@yuvalpinter/attention-is-not-not-explanation-dbc25b534017)

- [ ] [Seq2Seq - Nathan Lintz Sequence Modeling With Neural Networks](https://indico.io/blog/sequence-modeling-neuralnets-part1/) <br />
    Using of Seq2Seq: Since the decoder model sees an encoded representation of the input sequence as well as the translation sequence, it can make more intelligent predictions about future words based on the current word. For example, in a standard language model, we might see the word “crane” and not be sure if the next word should be about the bird or heavy machinery. However, if we also pass an encoder context, the decoder might realize that the input sequence was about construction, not flying animals. Given the context, the decoder can choose the appropriate next word and provide more accurate translations. <br />
    Without attention: Unfortunately, compressing an entire input sequence into a single fixed vector tends to be quite challenging. And, the context is biased towards the end of the encoder sequence, and might miss important information at the start of the sequence. <br />
    This mechanism will hold onto all states from the encoder and give the decoder a weighted average of the encoder states for each element of the decoder sequence. Now, the decoder can take “glimpses” into the encoder sequence to figure out which element it should output next. Our decoder network can now use different portions of the encoder sequence as context while it’s processing the decoder sequence, instead of using a single fixed representation of the input sequence. This allows the network to focus on the most important parts of the input sequence instead of the whole input sequence, therefore producing smarter predictions for the next word in the decoder sequence. Helps in better backpropagation to diff encoder states

- [ ] [The Transformer – Attention is all you need](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XnBcmZNKhTY)
- [ ] [Transformer Google Blog](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
    Need for transformer: Recurrent models due to sequential nature (computations focused on the position of symbol in input and output) are not allowing for parallelization along training, thus have a problem with learning long-term dependencies from memory. <br />
    Constraint of sequential computation: attempted by CNN models. However, in those CNN-based approaches, the number of calculations in parallel computation of the hidden representation, for input→output position in sequence, grows with the distance between those positions. The complexity of O(n) for ConvS2S and O(nlogn) for ByteNet makes it harder to learn dependencies on distant positions.<br />
    Transformer reduces the number of sequential operations to relate two symbols from input/output sequences to a constant O(1) number of operations. Transformer achieves this with the multi-head attention mechanism that allows to model dependencies regardless of their distance in input or output sentence.<br />
    The novel approach of Transformer is however, to eliminate recurrence completely and replace it with attention to handle the dependencies between input and output. The Transformer moves the sweet spot of current ideas toward attention entirely. It eliminates the not only recurrence but also convolution in favor of applying self-attention (a.k.a intra-attention). Additionally Transformer gives more space for parallelization. Transformer is claimed by authors to be the first to rely entirely on self-attention to compute representations of input and output. The encoder-decoder model is designed at its each step to be auto-regressive - i.e. use previously generated symbols as extra input while generating next symbol. Thus, xi+yi−1→yi <br />
    In each step, it applies a self-attention mechanism which directly models relationships between all words in a sentence, regardless of their respective position. In the earlier example “I arrived at the bank after crossing the river”, to determine that the word “bank” refers to the shore of a river and not a financial institution, the Transformer can learn to immediately attend to the word “river” and make this decision in a single step.  <br />
 
    Positional emb: In paper authors have decided on fixed variant using sin and cos functions to enable the network to learn information about tokens relative positions to the sequence. Of course authors motivate the use of sinusoidal functions due to enabling model to generalize to sequences longer than ones encountered during training.<br />

    Transformer reduces the number of operations required to relate (especially distant) positions in input and output sequence to a O(1). However, this comes at cost of reduced effective resolution because of averaging attention-weighted positions. To reduce this cost authors propose the multi-head attention. <br />
    
    Self-attention: In encoder, self-attention layers process input queries,keys and values that comes form same place i.e. the output of previous layer in encoder. Each position in encoder can attend to all positions from previous layer of the encoder. <br />
    
    In encoder phase (shown in the Figure 1.), transformer first generates initial representation/embedding for each word in input sentence (empty circle). Next, for each word, self-attention aggregates information form all other words in context of sentence, and creates new representation (filled circles). The process is repeated for each word in sentence. Successively building new representations, based on previous ones is repeated multiple times and in parallel for each word (next layers of filled circles). <br />

    Decoder acts similarly generating one word at a time in a left-to-right-pattern. It attends to previously generated words of decoder and final representation of encoder.<br />

- [ ] [Reformer Google Blog](https://ai.googleblog.com/2020/01/reformer-efficient-transformer.html)
- [ ] [Reformer Blog](https://www.pragmatic.ml/reformer-deep-dive/)
 focusing primarily on how the self-attention operation scales with sequence length, and proposing an alternative attention mechanism to incorporate information from much longer contexts into language models.<br />
 
- [ ] [BERT Google Blog](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)
    deeply bidirectional vs ELMO (shallow way to plugging two representations together)<br />
    Why does this matter? Pre-trained representations can either be context-free or contextual, and contextual representations can further be unidirectional or bidirectional. Context-free models such as word2vec or GloVe generate a single word embedding representation for each word in the vocabulary. For example, the word “bank” would have the same context-free representation in “bank account” and “bank of the river.” Contextual models instead generate a representation of each word that is based on the other words in the sentence. For example, in the sentence “I accessed the bank account,” a unidirectional contextual model would represent “bank” based on “I accessed the” but not “account.” However, BERT represents “bank” using both its previous and next context — “I accessed the ... account” — starting from the very bottom of a deep neural network, making it deeply bidirectional.<br />

- [ ] [GPT-2 OpenAI Blog](https://openai.com/blog/better-language-models/)
- [ ] [GPT-2 Blog](http://jalammar.github.io/illustrated-gpt2/)

- [ ] [ELECTRA Google Blog](https://ai.googleblog.com/2020/03/more-efficient-nlp-model-pre-training.html)

- [ ] [Self-supervised learning](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html)

- [ ] [Autoregressive Models](https://eigenfoo.xyz/deep-autoregressive-models/)
    At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.

- [ ] [A Survey of Long-Term Context in Transformers](https://www.pragmatic.ml/a-survey-of-methods-for-incorporating-long-term-context/)

## Resources

- [ ] [NLP Paper Summaries](https://github.com/dair-ai/nlp_paper_summaries)

- [ ] [NLP Progress](http://nlpprogress.com/)

- [ ] [How to a successful PhD student](https://people.cs.umass.edu/~wallach/how_to_be_a_successful_phd_student.pdf#page11)

- [ ] [Organizing files](http://www.cs.jhu.edu/~jason/advice/how-to-organize-your-files.html)

- [ ] [Writing code for NLP Research](https://github.com/allenai/writing-code-for-nlp-research-emnlp2018/blob/master/writing_code_for_nlp_research.pdf)

- [ ] [Missing Semester (MIT)](https://www.youtube.com/playlist?list=PLyzOVJj3bHQuloKGG59rS43e29ro7I57J)

## Topics
Self Supervised Learning <br />
SVM, Kernel and kernel Functions <br />
K-means, PCA, SVD <br />
Bagging Boosting <br />
Feature Selection <br />
Model Selection <br />
Optimization Algorithms <br />
HMM <br />
Transformer <br />
Active Learning <br />
Dependency Parsing <br />
POS tagging <br />
AdaBoast, AdaGrad, Ensembles: Check ML/NLP whatsapp group <br />
Random Forests <br />

