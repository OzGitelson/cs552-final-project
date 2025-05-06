# Hidden State Noise Injection as a Creativity Mechanism for LLMs
Large Language Models (LLMs) have achieved remarkable fluency in natural language generation, yet often default to safe and repetitive outputs, limiting their creative potential. In this work, we investigate whether deliberately injecting noise into the generative process can enhance the creativity and diversity of LLM-generated text, particularly in poetic domains. Using first lines from Walt Whitman’s Leaves of Grass as prompts, we systematically explore three noise injection strategies using a pretrained GPT-2 model: temperature scaling, Gaussian noise added to final logits, and Gaussian noise injected into hidden states between self-attention blocks. Our results reveal clear differences in their effects. Increasing temperature boost diversity but severely degrade coherence, while logit noise addition has minimal impact on performance. In contrast, hidden state noise injection results in more diverse and lexically rich outputs without sacrificing fluency. This suggests that targeted hidden state perturbations can serve as a viable mechanism for fostering creative generation in LLMs without compromising accuracy or coherence.

## To generate dataset from raw html
`python dataset.py`

## To run model and generate performance statistics
`python main,py`

## To display outputs and statistics for a given condition
`python compare_results.py`

## To display most fluent responses for a given condition
`python display_fluent_responses.py`
