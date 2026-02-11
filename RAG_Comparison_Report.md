# RAG System Comparison Report
*Generated: 2026-02-11 13:21*

## Overview

| System | Description |
|--------|-------------|
| **Vanilla RAG** | Simple retrieve-and-generate. Sends the raw user query directly to MedlinePlus search, chunks results, embeds into Chroma, retrieves + cross-encoder reranks, then prompts GPT-4o-mini. No query preprocessing or topic extraction. |
| **Query Expansion RAG** | Adds topic extraction (splits multi-topic queries), alias mapping (medical synonyms → MedlinePlus slugs), relevance-scored search ranking, and multi-topic combined search. Same vector store + reranking pipeline. |
| **Agentic RAG** | ReAct agent (LangChain `create_agent`) with tool use. Can decompose complex queries, perform multi-hop sequential searches, cache results, and synthesize across tool calls. Uses the same vector filtering + cross-encoder per tool call. |

---

## Simple (single-topic factual)
**Query:** *"What is hypertension?"*

### Performance

| System | Response Time | Word Count |
|--------|:------------:|:----------:|
| Vanilla RAG | 13.79s | 457 |
| Query Expansion RAG | 20.76s | 634 |
| Agentic RAG | 8.56s | 197 |

### Vanilla RAG Response

> ### What is Hypertension?
> 
> Hypertension, commonly known as high blood pressure, is a medical condition characterized by consistently elevated pressure in the arteries. It is a significant health concern because it can lead to serious complications, including heart attack and stroke.
> 
> #### Understanding Blood Pressure
> 
> - **Definition**: Blood pressure is the force exerted by circulating blood against the walls of the arteries. It is measured in millimeters of mercury (mmHg) and is expressed with two numbers:
>   - **Systolic Pressure**: The pressure in the arteries when the heart beats (highest pressure).
>   - **Diastolic Pressure**: The pressure in the arteries when the heart is at rest between beats (lowest pressure).
> 
> #### Symptoms of Hypertension
> 
> - **Often Asymptomatic**: Many individuals with hypertension do not experience noticeable symptoms, which is why it is often referred to as a "silent killer."
> - **Possible Symptoms**: In some cases, people may experience headaches, shortness of breath, or nosebleeds, but these symptoms typically occur only when blood pressure reaches dangerously high levels.
> 
> #### Causes of Hypertension
> 
> - **Primary Hypertension**: This type has no identifiable cause and develops gradually over many years.
> - **Secondary Hypertension**: This type is caused by an underlying condition, such as:
>   - Kidney disease
>   - Hormonal disorders
>   - Certain medications
>   - Sleep apnea
> 
> #### Treatment Options
> 
> - **Lifestyle Changes**: 
>   - **Diet**: Adopt a heart-healthy diet, such as the DASH (Dietary Approaches to Stop Hypertension) diet, which emphasizes fruits, vegetables, whole grains, and low-fat dairy.
>   - **Exercise**: Regular physical activity can help lower blood pressure.
>   - **Weight Management**: Maintaining a healthy weight is crucial for managing blood pressure.
>   - **Limit Alcohol and Tobacco**: Reducing alcohol intake and quitting smoking can significantly improve blood pressure levels.
>   
> - **Medications**: If lifestyle changes are **[...truncated]**

### Query Expansion RAG Response

> ### Understanding Hypertension
> 
> **Definition of Hypertension:**
> Hypertension, commonly known as high blood pressure, is a condition where the force of the blood against the walls of the arteries is consistently too high. Blood pressure is measured using two numbers: systolic pressure (the pressure when the heart beats) and diastolic pressure (the pressure when the heart is at rest). A typical blood pressure reading is expressed as systolic over diastolic (e.g., 120/80 mmHg).
> 
> ### Types of Hypertension
> 
> 1. **Primary (Essential) Hypertension:**
>    - **Description:** This is the most common type of hypertension, accounting for about 90-95% of cases. It develops gradually over many years and has no identifiable cause.
>    - **Risk Factors:** Age, genetics, obesity, lack of physical activity, high salt intake, and excessive alcohol consumption.
> 
> 2. **Secondary Hypertension:**
>    - **Description:** This type is caused by an underlying condition, such as kidney disease, hormonal disorders, or certain medications. It tends to appear suddenly and cause higher blood pressure than primary hypertension.
>    - **Common Causes:** 
>      - Kidney disease
>      - Adrenal gland tumors
>      - Certain congenital heart defects
>      - Medications (e.g., birth control pills, cold remedies)
> 
> ### Symptoms of Hypertension
> 
> - **Asymptomatic Nature:** Hypertension is often referred to as a "silent killer" because it usually has no symptoms. Many individuals may not realize they have high blood pressure until it is measured during a routine check-up.
> - **Potential Symptoms:** In severe cases, symptoms may include headaches, shortness of breath, or nosebleeds, but these are not common and typically occur only when blood pressure reaches dangerously high levels.
> 
> ### Diagnosis of Hypertension
> 
> - **Blood Pressure Measurement:** Diagnosis is made through regular blood pressure checks by a healthcare provider. This involves taking multiple readings at different times to confirm consistent high blood **[...truncated]**

### Agentic RAG Response

> Hypertension, commonly known as high blood pressure, is a condition where the force of the blood against the walls of the arteries is consistently too high. Blood pressure is measured using two numbers: 
> 
> 1. **Systolic Pressure**: This is the pressure in your arteries when your heart beats (the higher number).
> 2. **Diastolic Pressure**: This is the pressure in your arteries when your heart is resting between beats (the lower number).
> 
> For example, a blood pressure reading of 120/80 mmHg indicates a systolic pressure of 120 and a diastolic pressure of 80.
> 
> ### Diagnosis
> Hypertension often has no noticeable symptoms, which is why regular blood pressure checks are essential. A healthcare provider will typically take multiple readings at different appointments to confirm a diagnosis.
> 
> ### Types of Hypertension
> There are two main types of high blood pressure:
> 1. **Primary (Essential) Hypertension**: This type develops gradually over many years and has no identifiable cause.
> 2. **Secondary Hypertension**: This type is caused by an underlying condition, such as kidney disease or hormonal disorders.
> 
> For more detailed information, you can visit the [MedlinePlus page on High Blood Pressure](https://medlineplus.gov/highbloodpressure.html). 
> 
> If you have further questions or need specific information, feel free to ask!

---

## Moderate (actionable lifestyle advice)
**Query:** *"How can I lower cholesterol?"*

### Performance

| System | Response Time | Word Count |
|--------|:------------:|:----------:|
| Vanilla RAG | 12.53s | 436 |
| Query Expansion RAG | 16.31s | 516 |
| Agentic RAG | 13.57s | 337 |

### Vanilla RAG Response

> ### How to Lower Cholesterol
> 
> Lowering cholesterol is essential for maintaining heart health and reducing the risk of coronary artery disease and other heart-related issues. Here’s a detailed guide on how to effectively lower cholesterol levels through lifestyle changes and dietary modifications.
> 
> #### 1. **Heart-Healthy Lifestyle Changes**
> 
> - **Healthy Eating:**
>   - **DASH Eating Plan:** This dietary approach emphasizes fruits, vegetables, whole grains, and lean proteins while reducing saturated fats and cholesterol.
>   - **Therapeutic Lifestyle Changes (TLC) Diet:** This diet focuses on reducing saturated fat intake to less than 7% of total daily calories and increasing soluble fiber intake.
> 
> - **Weight Management:**
>   - Achieving and maintaining a healthy weight can significantly impact cholesterol levels. Even a modest weight loss of 5-10% can help lower LDL (bad cholesterol) levels.
> 
> - **Regular Physical Activity:**
>   - Engaging in regular exercise can help raise HDL (good cholesterol) levels while lowering LDL cholesterol. Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity each week.
> 
> #### 2. **Understanding Cholesterol Types**
> 
> - **LDL (Low-Density Lipoprotein):** Often referred to as "bad" cholesterol, high levels of LDL can lead to cholesterol buildup in arteries, increasing the risk of heart disease.
> - **HDL (High-Density Lipoprotein):** Known as "good" cholesterol, HDL helps transport cholesterol back to the liver for removal from the body. Higher levels of HDL are beneficial.
> 
> #### 3. **Dietary Recommendations**
> 
> - **Increase Soluble Fiber:**
>   - Foods high in soluble fiber, such as oats, beans, lentils, fruits, and vegetables, can help lower LDL cholesterol.
> 
> - **Healthy Fats:**
>   - Replace saturated fats found in red meat and full-fat dairy products with healthier fats from sources like olive oil, avocados, and nuts.
> 
> - **Limit Trans Fats:**
>   - Avoid trans fats found in many fried foods and commercial baked **[...truncated]**

### Query Expansion RAG Response

> To lower cholesterol effectively, it is essential to understand the types of cholesterol, the risks associated with high levels, and the strategies available for management. Below is a detailed overview of how to lower cholesterol, including lifestyle changes and treatment options.
> 
> ### Understanding Cholesterol
> 
> - **Types of Cholesterol**:
>   - **LDL (Low-Density Lipoprotein)**: Often referred to as "bad" cholesterol, high levels of LDL can lead to a buildup of cholesterol in the arteries, increasing the risk of coronary artery disease and other heart diseases.
>   - **HDL (High-Density Lipoprotein)**: Known as "good" cholesterol, HDL helps carry cholesterol away from the arteries back to the liver, where it can be removed from the body. Higher levels of HDL are beneficial.
> 
> ### Main Treatments for High Cholesterol
> 
> 1. **Lifestyle Changes**:
>    - **Dietary Adjustments**:
>      - **Therapeutic Lifestyle Changes (TLC) Diet**: Focuses on reducing saturated fats and cholesterol intake while increasing soluble fiber and plant sterols.
>      - **DASH Eating Plan**: Emphasizes fruits, vegetables, whole grains, and lean proteins while reducing sodium intake.
>    - **Weight Management**: Achieving and maintaining a healthy weight can significantly impact cholesterol levels. Even a modest weight loss can help lower LDL levels.
>    - **Physical Activity**: Regular exercise can help raise HDL cholesterol and lower LDL cholesterol. Aim for at least 150 minutes of moderate aerobic activity each week.
>    - **Avoiding Tobacco Smoke**: Quitting smoking can improve HDL cholesterol levels and benefit heart health overall.
>    - **Limiting Alcohol Intake**: If consumed, alcohol should be done in moderation, as excessive drinking can lead to higher cholesterol levels.
> 
> 2. **Medications**:
>    - If lifestyle changes alone are insufficient to lower cholesterol levels, healthcare providers may prescribe medications. Common classes of cholesterol-lowering medications include:
>      - **Statins**: **[...truncated]**

### Agentic RAG Response

> To lower cholesterol levels, particularly low-density lipoprotein (LDL) cholesterol, which is often referred to as "bad" cholesterol, you can adopt several lifestyle changes and dietary modifications. Here are some effective strategies:
> 
> ### 1. **Healthy Eating**
>    - **Reduce Saturated Fats**: Limit foods high in saturated fats, such as red meat and full-fat dairy products. Opt for lean meats and low-fat dairy instead.
>    - **Eliminate Trans Fats**: Avoid trans fats found in many fried foods and commercial baked products. Check food labels for "partially hydrogenated oils."
>    - **Increase Omega-3 Fatty Acids**: Include fatty fish (like salmon, mackerel, and sardines) in your diet, as they can help lower LDL cholesterol.
>    - **Eat More Fiber**: Soluble fiber, found in oats, beans, lentils, fruits, and vegetables, can help reduce cholesterol absorption in your bloodstream.
>    - **Add Plant Sterols and Stanols**: These substances, found in some fortified foods, can help block the absorption of cholesterol.
> 
> ### 2. **Regular Physical Activity**
>    - Aim for at least 150 minutes of moderate aerobic exercise each week, such as brisk walking, cycling, or swimming. Regular exercise can help raise high-density lipoprotein (HDL) cholesterol, the "good" cholesterol.
> 
> ### 3. **Maintain a Healthy Weight**
>    - Losing even a small amount of weight can help reduce cholesterol levels. Focus on a balanced diet and regular exercise to achieve and maintain a healthy weight.
> 
> ### 4. **Quit Smoking**
>    - If you smoke, quitting can improve your HDL cholesterol level. Additionally, the benefits of quitting smoking extend beyond cholesterol levels to overall heart health.
> 
> ### 5. **Limit Alcohol Intake**
>    - If you drink alcohol, do so in moderation. Excessive alcohol can lead to high cholesterol and other health issues.
> 
> ### 6. **Consider Medication**
>    - If lifestyle changes are not enough to lower your cholesterol, your healthcare provider may recommend medications such as **[...truncated]**

---

## Complex (multi-condition interaction)
**Query:** *"I have bipolar disorder and insomnia, what should I be aware of?"*

### Performance

| System | Response Time | Word Count |
|--------|:------------:|:----------:|
| Vanilla RAG | 1.38s | 18 |
| Query Expansion RAG | 14.51s | 550 |
| Agentic RAG | 16.45s | 337 |

### Vanilla RAG Response

> I couldn't find relevant information on MedlinePlus for your question. Please try rephrasing or consult a healthcare professional.

### Query Expansion RAG Response

> ### Understanding Bipolar Disorder and Insomnia
> 
> Both bipolar disorder and insomnia are significant health concerns that can impact an individual's quality of life. Understanding each condition and their potential interactions is crucial for effective management.
> 
> #### Bipolar Disorder
> 
> **Overview:**
> - Bipolar disorder is a mood disorder characterized by extreme mood swings, including manic episodes (elevated mood, increased energy) and depressive episodes (low mood, lack of energy).
> - Mood swings can affect behavior, energy levels, and activity levels.
> 
> **Symptoms:**
> - **Manic Episodes:** Increased energy, euphoria, irritability, decreased need for sleep, racing thoughts.
> - **Depressive Episodes:** Sadness, hopelessness, fatigue, changes in sleep patterns, difficulty concentrating.
> 
> **Causes:**
> - The exact cause is not fully understood but may involve genetic, biochemical, and environmental factors.
> 
> **Treatment Options:**
> - **Medications:** Mood stabilizers, antipsychotics, and antidepressants.
> - **Psychotherapy:** Cognitive-behavioral therapy (CBT) and other therapeutic approaches.
> - **Lifestyle Considerations:** Regular sleep patterns, stress management, and avoiding drugs and alcohol.
> 
> #### Insomnia
> 
> **Overview:**
> - Insomnia is a common sleep disorder where individuals have difficulty falling asleep, staying asleep, or achieving restful sleep.
> - It can be acute (short-term) or chronic (lasting more than a month).
> 
> **Symptoms:**
> - Difficulty falling asleep or staying asleep.
> - Waking up too early and not being able to go back to sleep.
> - Feeling unrefreshed upon waking.
> 
> **Causes:**
> - Acute insomnia may be triggered by stress, family pressures, or traumatic events.
> - Chronic insomnia is often secondary to other conditions, including medical issues, medications, or lifestyle factors such as caffeine and alcohol use.
> 
> **Treatment Options:**
> - **Cognitive Behavioral Therapy for Insomnia (CBT-I):** A structured program that helps address thoughts and behaviors that **[...truncated]**

### Agentic RAG Response

> When managing both bipolar disorder and insomnia, there are several important considerations to keep in mind:
> 
> ### Bipolar Disorder
> - **Mood Swings**: Bipolar disorder is characterized by intense mood swings, including manic episodes (elevated mood, increased energy) and depressive episodes (sadness, hopelessness). Understanding your specific type of bipolar disorder (Bipolar I, Bipolar II, or Cyclothymic disorder) is crucial for effective management.
> - **Treatment**: Treatment typically involves a combination of medication (mood stabilizers, antipsychotics) and psychotherapy. It's essential to work closely with a healthcare provider to find the right treatment plan.
> - **Triggers**: Stress, lack of sleep, and substance use can trigger mood episodes. Being aware of your triggers can help you manage your condition more effectively.
> 
> ### Insomnia
> - **Types of Insomnia**: Insomnia can be acute (short-term) or chronic (long-term). Chronic insomnia often results from underlying issues, including mental health conditions like bipolar disorder.
> - **Symptoms**: Insomnia may involve difficulty falling asleep, staying asleep, or waking up too early. Poor sleep quality can exacerbate mood swings and overall mental health.
> - **Management Strategies**: 
>   - **Sleep Hygiene**: Establish a regular sleep schedule, create a comfortable sleep environment, and avoid stimulants like caffeine and nicotine close to bedtime.
>   - **Relaxation Techniques**: Practices such as mindfulness, meditation, or gentle yoga can help reduce stress and promote better sleep.
>   - **Consult a Professional**: If insomnia persists, it may be beneficial to consult a healthcare provider for tailored strategies or medications that are safe to use alongside bipolar disorder treatments.
> 
> ### Overlapping Considerations
> - **Medication Interactions**: Some medications for bipolar disorder may affect sleep patterns. It's important to discuss any sleep issues with your healthcare provider, as they may need to adjust **[...truncated]**

---

## Very Complex (drug + comorbidity reasoning)
**Query:** *"I am on metformin for type 2 diabetes and also have chronic kidney disease. How might the drug's side effects affect my kidney condition?"*

### Performance

| System | Response Time | Word Count |
|--------|:------------:|:----------:|
| Vanilla RAG | 2.34s | 18 |
| Query Expansion RAG | 10.7s | 459 |
| Agentic RAG | 22.19s | 343 |

### Vanilla RAG Response

> I couldn't find relevant information on MedlinePlus for your question. Please try rephrasing or consult a healthcare professional.

### Query Expansion RAG Response

> ### Overview of Metformin and Its Use in Type 2 Diabetes
> 
> Metformin is a commonly prescribed medication for managing type 2 diabetes. It works primarily by improving insulin sensitivity and reducing glucose production in the liver, which helps lower blood sugar levels. While metformin is generally well-tolerated, it can have side effects, particularly in individuals with pre-existing kidney conditions.
> 
> ### Metformin and Chronic Kidney Disease (CKD)
> 
> #### Impact of CKD on Metformin Use
> - **Kidney Function**: Metformin is primarily excreted by the kidneys. In patients with chronic kidney disease (CKD), the ability to clear metformin from the body is reduced, which can lead to an accumulation of the drug in the bloodstream.
> - **Risk of Lactic Acidosis**: One of the most serious side effects associated with metformin, especially in patients with CKD, is lactic acidosis. This is a rare but potentially life-threatening condition that occurs when lactic acid builds up in the bloodstream faster than it can be removed. CKD can impair the kidneys' ability to eliminate lactic acid, increasing the risk of this complication.
> 
> #### Symptoms of Lactic Acidosis
> - Muscle pain or weakness
> - Difficulty breathing
> - Abdominal discomfort
> - Dizziness or feeling lightheaded
> - Slow or irregular heartbeat
> 
> If you experience any of these symptoms, it is crucial to seek medical attention immediately.
> 
> ### Management Considerations
> 
> #### Monitoring Kidney Function
> - **Regular Testing**: It is essential to have regular blood tests to monitor kidney function (e.g., serum creatinine and estimated glomerular filtration rate, or eGFR) while on metformin. This helps determine whether it is safe to continue the medication.
> - **Adjusting Dosage**: Depending on your kidney function, your healthcare provider may need to adjust the dosage of metformin or consider alternative medications for managing your diabetes.
> 
> #### Lifestyle Modifications
> - **Diet and Exercise**: Maintaining a healthy diet and **[...truncated]**

### Agentic RAG Response

> When taking metformin for type 2 diabetes while also having chronic kidney disease (CKD), there are several important considerations regarding the drug's side effects and its impact on kidney function:
> 
> ### Metformin and Kidney Function
> 1. **Kidney Monitoring**: Metformin is primarily eliminated by the kidneys. Therefore, if you have CKD, your healthcare provider will closely monitor your kidney function through regular blood tests. This is crucial because impaired kidney function can lead to an accumulation of metformin in the body, increasing the risk of side effects.
> 
> 2. **Risk of Lactic Acidosis**: One of the serious side effects of metformin is lactic acidosis, a rare but potentially life-threatening condition. This risk is heightened in individuals with significant kidney impairment, as the kidneys may not effectively clear lactic acid from the body. Symptoms of lactic acidosis include extreme tiredness, weakness, nausea, vomiting, stomach pain, and difficulty breathing.
> 
> 3. **Dosage Adjustments**: Depending on your level of kidney function, your doctor may need to adjust the dosage of metformin or consider alternative medications for managing your diabetes. It is essential to follow your healthcare provider's recommendations regarding medication adjustments.
> 
> ### Managing Chronic Kidney Disease
> - **Blood Sugar Control**: Keeping your blood sugar levels within the target range is vital for slowing the progression of CKD. Metformin can help achieve this, but it must be used cautiously in the context of kidney disease.
> - **Diet and Lifestyle**: Adopting a kidney-friendly diet, managing blood pressure, and maintaining a healthy weight can help protect your kidney function. Foods low in sodium, potassium, and phosphorus are often recommended for individuals with CKD.
> 
> ### Conclusion
> It's crucial to maintain open communication with your healthcare provider about your diabetes management and kidney health. Regular monitoring and adjustments to your treatment plan **[...truncated]**

---

## Comparative Analysis

### Critical Finding: Vanilla RAG Fails on Multi-Topic Queries

The most striking result is that **Vanilla RAG returned zero useful information** for both the Complex and Very Complex queries (18 words each -- just the fallback error message). Because it sends the raw natural-language query directly to MedlinePlus search without any topic extraction, queries like *"I have bipolar disorder and insomnia..."* fail to match any single MedlinePlus page. The 1-2 second response times for those queries reflect the system quickly giving up, not fast processing.

By contrast, **Query Expansion RAG** successfully answered all four queries by splitting multi-topic questions into individual searches (e.g., "bipolar disorder" and "insomnia" separately) and combining the results. **Agentic RAG** also answered all four, using its `decompose_query` tool to plan a multi-hop search strategy.

### Retrieval Quality (Observed)

| Capability | Vanilla RAG | Query Expansion RAG | Agentic RAG |
|------------|:-----------:|:-------------------:|:-----------:|
| Single-topic lookup ("hypertension") | Good (457 words, well-structured) | Good (634 words, most detailed) | Good (197 words, concise) |
| Actionable advice ("lower cholesterol") | Good (436 words) | Good (516 words) | Good (337 words) |
| Multi-condition ("bipolar + insomnia") | **FAILED** (no results) | Strong (550 words, covers both + overlap) | Strong (337 words, "Overlapping Considerations" section) |
| Drug-comorbidity ("metformin + CKD") | **FAILED** (no results) | Strong (459 words, lactic acidosis risk, monitoring) | Strong (343 words, multi-hop reasoning on interactions) |

### Response Depth Comparison

For the queries that all three systems answered successfully:

| Aspect | Vanilla RAG | Query Expansion RAG | Agentic RAG |
|--------|:-----------:|:-------------------:|:-----------:|
| Average word count (simple+moderate) | ~447 | ~575 | ~267 |
| Structural detail | Good headings, bullet points | Most detailed with sub-sections | Concise, focused sections |
| Source citations | MedlinePlus links appended | MedlinePlus links appended | Inline links within response |

Query Expansion RAG consistently produced the **longest and most detailed** responses. Agentic RAG produced **shorter but more focused** answers, likely because the agent selectively filters content through its tool calls rather than passing all retrieved context to the LLM.

### Response Time Trends

| Query Complexity | Vanilla RAG | Query Expansion RAG | Agentic RAG |
|-----------------|:-----------:|:-------------------:|:-----------:|
| Simple | 13.8s | 20.8s | 8.6s |
| Moderate | 12.5s | 16.3s | 13.6s |
| Complex | 1.4s (FAILED) | 14.5s | 16.5s |
| Very Complex | 2.3s (FAILED) | 10.7s | 22.2s |

- Vanilla RAG's fast times on complex queries are **misleading** -- it returned nothing useful.
- Query Expansion RAG is consistently in the 10-21s range regardless of complexity.
- Agentic RAG scales with complexity: fast for simple lookups (8.6s), slowest for multi-hop reasoning (22.2s) due to sequential tool calls and LLM planning steps.

### When to Use Each

| Scenario | Best System | Why |
|----------|-------------|-----|
| Simple factual lookups ("What is X?") | Any of the three | All perform well; Agentic was fastest here |
| Lifestyle / treatment advice (single condition) | Query Expansion RAG | Most detailed responses, reliable synonym handling |
| Multi-condition questions | Query Expansion or Agentic RAG | Vanilla RAG **cannot handle these at all** |
| Drug interaction / comorbidity reasoning | Agentic RAG | Decomposes into strategic multi-hop searches, synthesizes across findings |
| Latency-sensitive applications | Query Expansion RAG | Most predictable timing across all complexity levels |

### Key Takeaways

1. **Vanilla RAG has a hard ceiling.** It works well for straightforward single-topic queries but **completely breaks** when the question involves multiple medical concepts. This is a fundamental limitation of sending raw user queries to search without preprocessing.

2. **Query Expansion RAG is the best all-rounder.** Topic extraction + alias mapping + combined search gives it reliable performance across all complexity levels. It produced the most detailed answers and maintained consistent response times.

3. **Agentic RAG shines on reasoning-heavy queries.** Its ability to decompose questions, plan search strategies, and synthesize across multiple tool results makes it the strongest choice for complex drug-interaction and comorbidity questions. The trade-off is higher latency on complex queries.

4. **The gap widens with complexity.** For simple queries all three systems are comparable. The real differentiation appears at the Complex and Very Complex levels, where architectural differences in retrieval strategy become decisive.