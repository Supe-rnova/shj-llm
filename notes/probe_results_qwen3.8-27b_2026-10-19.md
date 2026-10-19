# Contamination probes — qwen3.8-27b
**Date:** 2026-10-19
**Settings:** temperature 1.0, reasoning_effort=none (Groq)
**Probe A source:** 24-trial ceiling runs from the main study (Gemini used 64-trial pilot runs)

## Probe A — post-run strategy report

### Type 1
- run_id: `59c124f8-c9fc-4d0c-9bb8-c2a7a30f0e95`, seed 101001, 24 trials, accuracy 0.917
- mapping: `{"1": "banding", "2": "lustre", "3": "grain"}`
- flags: ['experiment']

```
Based on the data provided from trials 1 through 24, I analyzed the three properties (Lustre, Banded/Unbanded, and Particle Size) to determine the underlying classification rule.

### Step-by-Step Deduction

**1. Analysis of Properties vs. Groups**

Let's look at the correlation between each property and the correct group assignment across all 24 trials.

*   **Property 1: Banded vs. Unbanded**
    *   **Banded Samples:**
        *   Trial 1: Banded $\rightarrow$ Beta
        *   Trial 3: Banded $\rightarrow$ Beta
        *   Trial 5: Banded $\rightarrow$ Beta
        *   Trial 8: Banded $\rightarrow$ Beta
        *   Trial 11: Banded $\rightarrow$ Beta
        *   Trial 12: Banded $\rightarrow$ Beta
        *   Trial 15: Banded $\rightarrow$ Beta
        *   Trial 16: Banded $\rightarrow$ Beta
        *   Trial 17: Banded $\rightarrow$ Beta
        *   Trial 20: Banded $\rightarrow$ Beta
        *   Trial 22: Banded $\rightarrow$ Beta
        *   Trial 24: Banded $\rightarrow$ Beta
        *   *Result: All Banded samples were Group Beta.*
    *   **Unbanded Samples:**
        *   Trial 2: Unbanded $\rightarrow$ Alpha
        *   Trial 4: Unbanded $\rightarrow$ Alpha
        *   Trial 6: Unbanded $\rightarrow$ Alpha
        *   Trial 7: Unbanded $\rightarrow$ Alpha
        *   Trial 9: Unbanded $\rightarrow$ Alpha
        *   Trial 10: Unbanded $\rightarrow$ Alpha
        *   Trial 13: Unbanded $\rightarrow$ Alpha
        *   Trial 14: Unbanded $\rightarrow$ Alpha
        *   Trial 18: Unbanded $\rightarrow$ Alpha
        *   Trial 19: Unbanded $\rightarrow$ Alpha
        *   Trial 21: Unbanded $\rightarrow$ Alpha
        *   Trial 23: Unbanded $\rightarrow$ Alpha
        *   *Result: All Unbanded samples were Group Alpha.*

*   **Property 2: Glassy vs. Dull (Lustre)**
    *   This property shows **no distinct correlation** with the group.
    *   Both Glassy and Dull samples appear in both Alpha and Beta groups.
    *   *Example:* Trial 1 (Glassy) is Beta; Trial 5 (Dull) is Beta.
    *   *Example:* Trial 2 (Dull) is Alpha; Trial 4 (Glassy) is Alpha.

*   **Property 3: Fine vs. Coarse (Particle Size)**
    *   This property also shows **no distinct correlation** with the group.
    *   Both Fine and Coarse samples appear in both Alpha and Beta groups.
    *   *Example:* Trial 1 (Fine) is Beta; Trial 3 (Coarse) is Beta.
    *   *Example:* Trial 2 (Fine) is Alpha; Trial 7 (Coarse) is Alpha.

**2. Identifying Distractors**

In cognitive science experiments like this (often based on Reed's concept learning research), the "Lustre" and "Particle Size" are considered **irrelevant features** or distractors. The "Banded/Unbanded" feature is the **relevant feature** that determines the category.

### Final Rule

The strategy
```

### Type 6
- run_id: `d407a1c5-b970-496b-8cea-8d79466b1b4a`, seed 106001, 24 trials, accuracy 0.75
- mapping: `{"1": "grain", "2": "banding", "3": "lustre"}`
- flags: ['experiment']

```
Based on the data provided, we can deduce the correct classification rule by analyzing the properties of the samples across the trials. The three properties involved are **Texture** (glassy vs. dull), **Bandaging** (banded vs. unbanded), and **Grain Size** (fine vs. coarse).

Let's list the correct groupings for every unique combination of properties seen in the experiment:

1.  **Glassy, Banded, Coarse**:
    *   Trial 1: Group Beta
    *   Trial 10: Group Beta
    *   Trial 23: Group Beta
    *   *Result:* Always **Group Beta**

2.  **Dull, Unbanded, Fine**:
    *   Trial 2: Group Alpha
    *   Trial 11: Group Alpha
    *   Trial 17: Group Alpha
    *   *Result:* Always **Group Alpha**

3.  **Glassy, Unbanded, Coarse**:
    *   Trial 3: Group Alpha
    *   Trial 9: Group Alpha
    *   Trial 21: Group Alpha
    *   *Result:* Always **Group Alpha**

4.  **Dull, Unbanded, Coarse**:
    *   Trial 4: Group Beta
    *   Trial 15: Group Beta
    *   Trial 22: Group Beta
    *   *Result:* Always **Group Band**

5.  **Glassy, Banded, Fine**:
    *   Trial 5: Group Alpha
    *   Trial 12: Group Alpha
    *   Trial 20: Group Alpha
    *   *Result:* Always **Group Alpha**

6.  **Dull, Banded, Coarse**:
    *   Trial 6: Group Alpha
    *   Trial 16: Group Alpha
    *   Trial 18: Group Alpha
    *   *Result:* Always **Group Alpha**

7.  **Glassy, Unbanded, Fine**:
    *   Trial 7: Group Beta
    *   Trial 14: Group Beta
    *   Trial 24: Group Beta
    *   *Result:* Always **Group Beta**

8.  **Dull, Banded, Fine**:
    *   Trial 8: Group Beta
    *   Trial 13: Group Beta
    *   Trial 19: Group Beta
    *   *Result:* Always **Group Beta**

### Analysis of the Rule

Let's look for a single property or combination that determines the group.

*   **Is it based on Texture?**
    *   Glassy samples belong to both Alpha (e.g., Trial 5, 9) and Beta (e.g., Trial 1, 7).
    *   Dull samples belong to both Alpha (e.g., Trial 2, 6) and Beta (e.g., Trial 4, 8).
    *   *Conclusion:* No.

*   **Is it based on Banding?**
    *   Banded samples belong to both Alpha (e.g., Trial 5, 6) and Beta (e.g., Trial 1, 8).
    *   Unbanded samples belong to both Alpha (e.g., Trial 2, 3) and Beta (e.g., Trial 4, 7).
    *   *Conclusion:* No.

*   **Is it based on Grain Size?**
    *   Fine samples belong to both Alpha (e.g., Trial 2, 5) and Beta (e.g., Trial 7, 8).
    *   Coarse samples belong
```

## Probe B — paradigm recognition (fresh context)

### Type 1
- flags: ['xor', 'categorization', 'cognitive psychology', 'psychology', 'experiment']

```
Based on the data provided, the classification rule is entirely determined by the **first feature** (Input 1).

*   If the first feature is **1**, the output is **A**.
*   If the first feature is **0**, the output is **B**.

The other two features (Input 2 and Input 3) are **irrelevant noise** and have no bearing on the classification.

### Task Identification

This specific structure—where one feature is predictive and the others are uninformative (irrelevant)—is the classic setup for testing **Feature-Based Attention** or **Selective Attention** in cognitive psychology. It is often used in:
1.  **Perceptual Learning** experiments.
2.  **Attention Mechanism** studies (e.g., Stroop-like tasks, visual search, or categorization tasks).
3.  **Machine Learning Benchmarks** for testing **Feature Selection** or **Regularization** (to see if a model can correctly identify the relevant feature and ignore the irrelevant ones).

However, because the data is **linearly separable** and the rule is **deterministic and simple**, it is most specifically recognized as:

**The "Relevant Feature" vs. "Irrelevant Feature" Paradigm**

In experimental psychology and cognitive science, this is a standard control condition to force participants (or models) to rely on selective attention.

### Why it’s not other common categories?
*   **It is not the XOR problem:** XOR would be e.g., `00->B, 01->A, 10->A, 11->B`. Here, the rule is straightforward.
*   **It is not the 2/4 Concentration Task (D-Keystroke):** That is a motor/cognitive flexibility test (press 1 for 4s, press 2 for 2 and 3, etc.).
*   **It is not the N-back task:** That tests working memory span.
*   **It is not the Iowa Gambling Task:** That tests explicit vs. implicit learning of reward contingencies.

### Conclusion

This is a **Feature-Salience or Selective Attention** task, specifically a **deterministic binary categorization task** designed to isolate the influence of **one relevant feature** amidst **two irrelevant features**.

If this is from a machine learning context, it is a **linearly separable classification problem** used to test **model simplicity** and **overfitting resistance** (a good model should ignore features 2 and 3).
```

### Type 2
- flags: ['xor', 'cognitive psychology', 'psychology', 'experiment']

```
Based on the data provided, the rule governing the classification is:

**If the first bit is 0, the category is B. If the first bit is 1, the category is A.**

The other two bits are irrelevant to the classification.

This specific pattern corresponds to a simplified version of a **conjunction rule** or more commonly in psychological literature, a **single-feature rule** (also known as a **one-dimensional task**).

However, if we look at the structure of the data:
- The dataset is symmetrical and simple.
- It isolates **Feature 1** as the sole predictor.

### Most Specific Recognition: **The "One-Dimensional" or "Single-Feature" Concept Formation Task**

In cognitive psychology and experimental paradigms involving **Concept Formation** (e.g., the **Winning Strategies** experiment by Bruner, Goodnow, and Austin, 1956; or the **Human Concept Learning** tasks), this is a classic example of a **rule that depends on only one attribute** when three are offered.

Participants are typically presented with a set of items (like these) and must infer the rule that separates the "positive" examples (Category A) from the "negative" examples (Category B).

### Why this is a specific paradigm element:
1.  **Rule Simplicity**: Many concept learning experiments control for the "complexity" of the rule. A rule based on a single feature is the simplest possible rule in a multi-attribute space.
2.  **Irrelevant Features**: The second and third bits are perfectly balanced (no correlation with the outcome) to test if participants get distracted by "red herring" features.
3.  **Binary Classification**: It fits the binary classification framework used in **Artificial Concept Learning** studies.

### Alternative Interpretation (Neural Networks/Machine Learning):
In the context of early **Perceptron** experiments (Minsky's famous proof of the **XOR problem**), researchers often tested various patterns to show what *can* and *cannot* be solved.
- This task **is** solvable by a simple Perceptron because it is linearly separable.
- The XOR problem (which is *not* linearly separable) has the pattern: 00->0, 01->1, 10->1, 11->0.
- Your data is **not** XOR. It is a **linear threshold** problem.

### Conclusion:
The most accurate specific label for this experimental item set is:

> **A single-attribute rule in a multi-attribute concept learning task (e.g., from the Artificial Concept Learning paradigm).**

If you are looking for a named specific puzzle or brain teaser, it is often used in **introductory psychology** or **cognitive science** textbooks to introduce **concept formation** and **hypothesis testing**, demonstrating that people initially often test **conjunctions** (e.g., "all bits are 1" or "first bit is 1 AND last bit is 0") rather than the simpler disjunction or single-feature rule, a phenomenon related to **positive asymmetry** in hypothesis testing.

But fundamentally, it is a **linearly separable binary classification task** designed to test **rule induction**.
```

### Type 6
- flags: ['xor', 'exclusive or', 'parity']

```
This pattern is the **Wason 4-Card Problem** (specifically, a version designed to test logical reasoning vs. intuitive heuristic processing).

### Step-by-Step Deduction

1.  **Analyze the Structure**:
    The problem presents a conditional rule: "If an item has Feature X, then it belongs to Category Y." However, the rule is not explicitly stated in text; instead, it is hidden within the data mapping.
    
    Let's deduce the underlying logical rule from the provided mappings:
    *   `000 -> B`
    *   `001 -> A`
    *   `010 -> A`
    *   `011 -> B`
    *   `100 -> A`
    *   `101 -> B`
    *   `110 -> B`
    *   `111 -> A`

    Let the three binary features be $F_1, F_2, F_3$.
    Let's look for a conditional rule of the form: **If [Condition on First Feature] then [Category]**.
    
    *   When $F_1 = 0$ (Cases 000, 001, 010, 011):
        *   000 -> B
        *   001 -> A
        *   010 -> A
        *   011 -> B
        There is no consistent category.
    
    *   When $F_1 = 1$ (Cases 100, 101, 110, 111):
        *   100 -> A
        *   101 -> B
        *   110 -> B
        *   111 -> A
        There is no consistent category.

    Let's try a different angle. In classic logic puzzles of this type, the rule is often **"If the first bit is 1, then the last bit is 1"** or similar structural constraints. But here the output is A/B.
    
    Let's look at the standard **Wason Selection Task**. It usually involves 4 cards and a rule. Here we have 8 items (all possible combinations of 3 bits). This suggests a **logical classification** task where the participant must sort items into A or B based on a hidden rule.

    Let's re-examine the mapping to find the rule.
    Let's check the XOR (exclusive OR) of the three features.
    *   $0 \oplus 0 \oplus 0 = 0 \rightarrow B$
    *   $0 \oplus 0 \oplus 1 = 1 \rightarrow A$
    *   $0 \oplus 1 \oplus 0 = 1 \rightarrow A$
    *   $0 \oplus 1 \oplus 1 = 0 \rightarrow B$
    *   $1 \oplus 0 \oplus 0 = 1 \rightarrow A$
    *   $1 \oplus 0 \oplus 1 = 0 \rightarrow B$
    *   $1 \oplus 1 \oplus 0 = 0 \rightarrow B$
    *   $1 \oplus 1 \oplus 1 = 1 \rightarrow A$

    **The Rule**: If the XOR (parity) of the three bits is 1 (odd number of 1s), the category is **A**. If the XOR is 0 (even number of 1s, including 0), the category is **B**.
```

## Probe C — direct knowledge check (fresh context)

- flags: ['shepard', 'hovland', 'jenkins', 'category learning', 'categorization', 'prototype', 'cognitive psychology', 'psychology', 'experiment', '1961']

```
### The Shepard, Hovland, and Jenkins (1961) Category Learning Task

The study conducted by Roger N. Shepard, C.I. Hovland, and Jack C. Jenkins in 1961 is a landmark experiment in cognitive psychology that investigated how humans learn to categorize stimuli. They developed a control task designed to isolate the **conceptual structure** of a task from the difficulty of the stimuli themselves.

#### 1. Structure of the Task
The task involves learning to map sets of stimuli (usually geometric shapes) into two distinct response classes (e.g., pressing button "A" or "B").

- **Stimuli**: Shapes are defined along three dimensions (features), each with two possible states:
  1. Shape (Circle or Square)
  2. Color (Red or Blue)
  3. Size (Large or Small)
- **Goal**: The subject must analyze the stimulus and respond based on which feature determines the "Positive" class.

Because there are 3 features, there are $2^3 = 8$ distinct stimuli (e.g., Large Red Circle, Small Blue Square).

---

### The Difficulty Ordering

The central finding of the study was that there is a specific, consistent hierarchy of difficulty in learning these concepts, regardless of the functional complexity of the rules.

The difficulty order (from **Easiest** to **Hardest**) is as follows:

1.  **Conjunction ($+$)**
2.  **Disjunction ($\left| \right|$)**
3.  **Negated Conjunction ($\left|\!\right|$)**
4.  **Two-attribute concept (Negated Disjunction)**
    *(Often grouped as a 3rd level or "Difficult" level, effectively sharing the 3rd rank with Negated Conjunction in the original data, but distinguished sometimes as "Difficult One-Dimension")*
5.  **Two-attribute concept (Conjunctive Double Negative)**
6.  **Identity / Homoscapes ($\equiv$)**

*(Note: In many modern summaries, the two 4-feature levels are consolidated, or the top level is called the "Polymorphic class.")*

#### Detailed Breakdown of the Concepts

Below is the logical definition of each concept in English, followed by the boolean logic (where features are $A, B, C$ with values 0 or 1, and the target class is logic 1).

| Rank | Concept Name | Logical Rule (English) | Boolean Logic | Difficulty Example |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Conjunction** | Response is 1 only when **One specific attribute** is present. | $A=1$ | "Respond if Red." (Size and Shape don't matter) |
| **2** | **Disjunction** | Response is 1 if **Either** of two specific attributes is present. | $A=1$ OR $B=1$ | "Respond if Red OR Large." |
| **3** | **Negated Conjunction** | The response is 1 if a **specific pair** is present, but if either part is missing, the other must be of a specific type. | $A=1$ AND $B=1$ | "Large Red: Yes. Small Red: No. Large Blue: No. Small Blue: No." |
| **4** | **Negated Disjunction** | Response is 1 if **Either** attribute is present, with a dependency. | $(A=1$ AND $B=0)$ <br>OR $(A=0$ AND $B=1)$ | (See details below) |
| **5** | **Two-attribute concept** | This is often the grouping for concept types that rely on the **absence** of specific items or specific pairings. | $A=1$ AND $B=1$ <br>OR $A=0$ AND $B=0$ | "Respond to the pair (Red, Circle) or the exact opposite (Blue, Square)." |
| **6** | **Identity (Relational)** | Response depends on the **relationship** between two attributes. | $A=B$ | "Respond if the color matches the shape" (e.g., Red Circle = 1, Blue Square = 1). |

### Summary of the Hierarchy Logic

The difficulty increases because the subject has to ignore more irrelevant information as they move down the list:

1.  **Easy (Conjunction & Disjunction)**: The correct response depends only on **one** feature (Conjunction) or is supported by **two** features (Disjunction). The subject can essentially look at one edge of the 3D cube and say "Yes."
2.  **Medium (Negated Conjunction)**: The correct response requires checking **two** features simultaneously. If one is present (e.g., Large), the other must also be present.
3.  **Hard (Identity)**: The response depends on the **correlation** between features. The subject must look at Attribute A, look at Attribute B, and compare them. This is cognitively the most difficult because it is not defined by the presence or absence of a specific shape/color, but by their **match** (or mismatch).

Shepard, Hovland, and Jenkins proposed that human categorization strategies do not perfectly align with algorithmic (mathematical) complexity (e.g., bit-counting), but rather align with a functional hierarchy where **simpler relations (identity)** are perceived as harder because they lack a "prototype" feature that drives
```
