# Contamination probes — gemini-3.1-flash-lite
**Date:** 2026-09-15
**Settings:** temperature 1.0, thinking_budget=0

## Probe A — post-run strategy report

### Type 1
- run_id: `8e75f3b4-1dff-4eae-bf00-ca0bf943b5ed`, seed 1001, accuracy 0.953
- mapping: `{"1": "banding", "2": "grain", "3": "lustre"}`
- flags: none

```
Based on the data provided throughout the 64 trials, the rule for categorizing the mineral samples is based entirely on the **"Banded" vs. "Unbanded"** property.

Here is the breakdown of the rule:

*   **Group Alpha:** Any mineral sample that is **Unbanded** belongs to Group Alpha. 
    *(This holds true regardless of whether the luster is glassy or dull, or whether the texture is coarse or fine.)*
*   **Group Beta:** Any mineral sample that is **Banded** belongs to Group Beta. 
    *(This also holds true regardless of whether the luster is glassy or dull, or whether the texture is coarse or fine.)*

Looking back at the trials:
*   Trial 1: Unbanded = Alpha (Correct)
*   Trial 3: Banded = Beta (Correct)
*   Trial 5: Banded = Beta (Correct)
*   Trial 6: Unbanded = Alpha (Correct)

The luster (glassy/dull) and texture (coarse/fine) are irrelevant distractors; the presence of banding is the sole deterministic factor for the category.
```

### Type 6
- run_id: `5c111b6f-c7af-4f07-b653-13e32e4fa248`, seed 6001, accuracy 0.922
- mapping: `{"1": "grain", "2": "lustre", "3": "banding"}`
- flags: ['xor', 'parity', 'categorization']

```
Based on the feedback provided throughout the 64 trials, I identified that the categorization is determined by a **parity rule** (the sum of the features). 

To solve this, I assigned a numerical value to each property:

*   **Luster:** Glassy = 0, Dull = 1
*   **Texture:** Banded = 0, Unbanded = 1
*   **Grain size:** Coarse = 0, Fine = 1

Using these values, I calculated the sum for each sample. The classification rule is as follows:

*   **If the sum of the features is even (0 or 2), the mineral belongs to Group Beta.**
*   **If the sum of the features is odd (1 or 3), the mineral belongs to Group Alpha.**

### Verification Examples:
*   **Trial 2:** Glassy (0) + Banded (0) + Fine (1) = 1 (Odd) → **Group Alpha** (Correct)
*   **Trial 3:** Dull (1) + Unbanded (1) + Coarse (0) = 2 (Even) → **Group Beta** (Correct)
*   **Trial 4:** Glassy (0) + Unbanded (1) + Coarse (0) = 1 (Odd) → **Group Alpha** (Correct)
*   **Trial 5:** Glassy (0) + Unbanded (1) + Fine (1) = 2 (Even) → **Group Beta** (Correct)
*   **Trial 6:** Dull (1) + Unbanded (1) + Fine (1) = 3 (Odd) → **Group Alpha** (Correct)

This "parity" or "XOR-like" strategy accounts for every trial provided in the sequence, consistently predicting the correct group based on whether the total count of "Dull," "Unbanded," and "Fine" features is even or odd.
```

## Probe B — paradigm recognition (fresh context)

### Type 1
- flags: ['xor', 'category learning', 'categorization', 'nosofsky', 'prototype', 'cognitive psychology', 'psychology', 'experiment']

```
The dataset you provided describes a classic **Concept Learning Task**, specifically illustrating a **Rule-Based Category Learning** experiment.

### The Specific Paradigm
This dataset corresponds to a **conjunctive rule** (or "Boolean rule") task, which is a staple in cognitive psychology, dating back to the seminal work of **Jerome Bruner, Jacqueline Goodnow, and George Austin (1956)** in their book *A Study of Thinking*.

In modern computational modeling of categorization, this specific set is often referred to as a **Type 1 problem** according to the taxonomy established by **F. Gregory Ashby and W. Todd Maddox**, and famously operationalized in the benchmark study by **Robert Nosofsky, Stephen Palmeri, and Stephanie McKinley (1994)**.

### Why this specific structure?
The features in your set are $\{x_1, x_2, x_3\}$. By looking at the mappings:
*   If $x_1 = 0$, the category is B.
*   If $x_1 = 1$, the category is A.

The category membership is determined **solely by the first feature**. This is known as a **single-dimension rule**. 

### Context in Cognitive Science
This specific structure is used to distinguish between different types of human learning strategies:

1.  **Rule-Based Learning (The "Hypothesis Testing" Strategy):** Humans learn this task very quickly. Because there is a perfect correlation between the first feature and the category label, participants typically use an explicit strategy: "If it starts with 1, it’s A; otherwise, it’s B."
2.  **Comparison to Complexity:** In the Ashby/Maddox taxonomy, this is the easiest type of category learning (Type 1). Other types include:
    *   **Type 2 (The Exclusive-OR/XOR problem):** Requires two rules (e.g., if $x_1 = x_2$, then A).
    *   **Type 6 (The "Family Resemblance" or "Non-linear" problem):** Where no simple rule exists, and participants must rely on implicit, similarity-based, or prototype-based learning.

### Summary
You are looking at a **simple conjunctive classification task**. In a laboratory setting, this would be used as a baseline to demonstrate **rule-induction**. It is the "control" condition used to contrast how the brain switches from explicit, logic-based hypothesis testing (which handles this task easily) to implicit, reinforcement-based learning (which is required for more complex, multi-dimensional structures).
```

### Type 2
- flags: ['shepard', 'hovland', 'jenkins', 'type ii', 'xor', 'exclusive or', 'category learning', 'exemplar', 'prototype', 'cognitive psychology', 'psychology', 'experiment', '1961']

```
This set of data is a classic example of an **XOR (Exclusive OR) logical function** in a classification context. 

Specifically, in cognitive psychology and machine learning, this is the **"XOR Problem,"** famously used to demonstrate the limitations of linear classifiers (like the Perceptron) and to illustrate **category learning** through rule-based or prototype-based systems.

### Identification
*   **Logical structure:** The classification follows the rule: **Category A if the middle bit and the first bit are different** (or more simply, if you XOR the first two bits: `0^1=1`, `1^0=1` for A, and `0^0=0`, `1^1=0` for B).
*   **The "XOR Problem":** In the 1969 book *Perceptrons*, Minsky and Papert proved that a single-layer neural network cannot solve this problem because the classes are **linearly non-separable**. You cannot draw a single straight line through a 2D or 3D space to separate the A's from the B's.

### Relevant Experimental Paradigms
1.  **Category Learning (Concept Learning):** This is a hallmark stimulus set for experiments on **"Rule-based vs. Information-integration"** category learning. Participants are presented with these items and must learn the mapping through trial and error. 
    *   Research often uses these to test if humans use **rule-based strategies** (e.g., "If the second bit is 1, it's A") or **similarity-based strategies** (e.g., comparing current items to stored exemplars of A and B).
2.  **Boolean Concept Learning:** Introduced by Shepard, Hovland, and Jenkins (1961), who defined six types of category structures based on three binary features. The XOR structure is known as **Type II**.
    *   **Type II (XOR) is significant** because it is famously more difficult for human learners to master than Type I (a simple unidimensional rule) but easier than Types III–VI. It represents a "configural" learning task where the value of one feature depends on the value of another.
3.  **Connectionism/Neural Network Training:** If you are looking at this in a computational context, this is the "Hello World" of multi-layer perceptrons (MLPs). It is the standard training set used to demonstrate that adding a **hidden layer** with a non-linear activation function allows a network to solve linearly non-separable problems.

**In summary:** You are looking at **Type II (XOR) concept learning**, which is the foundational paradigm for studying how humans and machines learn non-linear logical dependencies.
```

### Type 6
- flags: ['shepard', 'hovland', 'jenkins', 'type ii', 'type vi', 'xor', 'exclusive or', 'parity', 'category learning', 'prototype', 'cognitive psychology', 'psychology', 'experiment', '1961']

```
The set of items you have provided describes a classic **learning task based on the XOR (exclusive OR) parity problem**, specifically in the context of **Category Learning** or **Concept Formation**.

### 1. The Logic of the Set
The classification rule follows the **Parity function**: 
*   An item belongs to **Category A** if the number of 1s in the string is **odd** (1 or 3).
*   An item belongs to **Category B** if the number of 1s in the string is **even** (0 or 2).

### 2. The Experimental Paradigm
In cognitive psychology and machine learning, this is most commonly referred to as the **"XOR Problem" (or Parity Problem)** within **Concept Learning**.

*   **In Cognitive Psychology:** This is a classic "nonlinear" category learning task. Human participants typically find this task much harder to learn than tasks that are linearly separable (like those based on a single feature or a simple conjunctive rule). It is often used to demonstrate that human category learning is not merely an exercise in finding a linear boundary, but requires more complex rule-based or prototype-based processing.
*   **In Connectionism/Neural Networks:** This dataset is the "Hello World" of neural network limitations. In the 1960s, Minsky and Papert (in their book *Perceptrons*) famously pointed out that a single-layer perceptron cannot solve the XOR problem because it is not linearly separable in feature space. This observation led to the "AI Winter" until the development of backpropagation and multi-layer networks (which can solve this by creating hidden representations).

### 3. Specific Significance
*   **Rule Complexity:** This specific set belongs to a class of tasks known as **"Boolean Concept Learning."** The difficulty of these tasks for humans is often measured by the **"Boolean complexity"** of the rule (often calculated by the length of the shortest logical description required to define the category).
*   **Shepard, Hovland, and Jenkins (1961):** This specific structure is a variation of the classic experiments conducted by Shepard, Hovland, and Jenkins. They examined how people learn six different types of logical structures using binary-featured stimuli. While they typically used 3-feature stimuli to create 8 items, your set specifically maps to the **"Type II"** or **"Type VI"** problems depending on the dimensions, which are known to be the most cognitively taxing for human learners because they require a "non-linear" strategy.

**In summary:** You are looking at a **non-linearly separable concept learning task**. If you were to present this to human subjects, you would likely be studying **category induction, rule-based reasoning, or hypothesis testing strategies.**
```

## Probe C — direct knowledge check (fresh context)

- flags: ['shepard', 'hovland', 'jenkins', 'type ii', 'type vi', 'xor', 'parity', 'categorization', 'alcove', 'cognitive psychology', 'psychology', 'experiment', '1961']

```
The **Shepard, Hovland, and Jenkins (1961) task** is a seminal study in cognitive psychology that established the fundamental ways humans learn to categorize objects. It is often referred to as the "SHJ task" and serves as a benchmark for testing theories of human and machine learning.

### The Task Setup
In this experiment, participants were asked to learn to classify stimuli into two categories (Category A and Category B). The stimuli were geometric shapes that varied along **three binary dimensions**:
1.  **Size** (Large or Small)
2.  **Color** (Black or White)
3.  **Shape** (Square or Triangle)

Since there are 3 binary dimensions ($2^3$), there are a total of **8 unique stimuli**. The task involves assigning 4 of these 8 items to Category A and the remaining 4 to Category B.

### The Six Types of Learning
Shepard, Hovland, and Jenkins identified **six logically distinct ways** to partition these 8 stimuli into two groups of 4. These types are defined by the "rule" required to solve the categorization:

1.  **Type I (The Simplest):** Categorization is based on only **one dimension**. (e.g., "All white objects are A, all black objects are B"). You can ignore the other two dimensions entirely.
2.  **Type II (The Rule + Exception):** Categorization is based on one dimension, but with exceptions. This is logically equivalent to a **"Conditional Rule"** (e.g., "If size is large, then Category A; if size is small, then Category B, *unless* it's a small white triangle").
3.  **Type III:** Requires learning a relevant dimension plus a dependency on the other two. It involves a "logical conjunction" combined with an exception.
4.  **Type IV:** Requires complex, non-linear processing. It is similar in complexity to Type III.
5.  **Type V:** Also highly complex; requires integrating all three dimensions in a non-obvious way.
6.  **Type VI (The Hardest):** Known as the **"Parity Problem"** or "Exclusive-OR" (XOR) problem. Every single dimension is relevant, and you must know the value of all three dimensions to determine the category. No single dimension or pair of dimensions provides a shortcut.

---

### The Difficulty Ordering
Shepard, Hovland, and Jenkins found that the speed and accuracy with which humans learn these types follow a distinct hierarchy:

**Type I < Type II < Types III, IV, V < Type VI**

*   **Type I is the easiest:** Because it relies on a single, perfectly predictive feature.
*   **Type II is next:** It is harder than Type I because the rule is not perfectly consistent, but humans find it intuitive because we are naturally good at identifying "rules with exceptions."
*   **Types III, IV, and V are intermediate:** Performance drops significantly here because the rules become increasingly abstract and difficult to verbalize.
*   **Type VI is the hardest:** Participants struggle the most with this type because there are no simple rules; the category membership depends on a complex combination of all features (an XOR-like relationship).

### Why this matters
The SHJ task is significant because it demonstrated that **learning is not just about memorizing associations.** If learning were purely about memory, all six types would be equally difficult. Instead, the results prove that human cognition is biased toward **rule-based processing.** We prefer simple, single-dimensional rules (Type I) and struggle with high-dimensional interactions (Type VI). This experiment remains a primary testing ground for mathematical models of categorization, such as ALCOVE (Attention Learning Covering Map) and SUSTAIN.
```
