# PolicyPal — Manual Evaluation Set
**PDF tested:** Star Health Family Health Optima
**Date:** 2026-06-30

## Scoring
- ✅ Correct — answer matches policy document
- ⚠️ Partial — answer is close but missing details
- ❌ Wrong — hallucination or completely off

---

## Results

### Q1: What is the waiting period for pre-existing diseases?
**Expected:** [The waiting period for pre-existing diseases (PED) depends on the policy term: 30 months for a 3-year policy term.
36 months for 1-year and 2-year policy terms.]
**Actual:** [ "The waiting period for pre-existing diseases is 24 months from the date of inception of the policy (Page 11) and also 36 months for coverage under the policy after the expiry for any pre-existing disease, subject to it being declared at the time of application and accepted by the Insurer (Page 10). Additionally, pre-existing diseases are defined and have specific conditions on pages 4."]
**Retrieval correct?:**  Partial
**Score:** ⚠️ 
**Notes:** Retrieved 24 months and 36 months figures but confused them. 
Correct answer is 36 months (1/2 year policy) or 30 months (3 year policy). 
The 24 months figure likely came from a different waiting period clause 
(specific disease waiting period), not PED. Retrieval brought back wrong clause.

---

### Q2: What is the room rent limit per day?
**Expected:** ₹5 lakh	Up to 1% of the Sum Insured per day (i.e., ₹5,000/day). ₹10 lakh, ₹15 lakh, ₹20 lakh, ₹25 lakh	Any room, except Suite or a higher category. ₹50 lakh, ₹75 lakh, ₹1 crore, ₹2 crore	Any room (no room category restriction)
**Actual:** ["The room rent limit per day is up to 1% of the Sum Insured per day (Page 5 and Page 18)."]
**Retrieval correct?:** Partial - retrieved the 1% rule but missed the tiered breakdown by sum insured
**Score:**  ⚠️ 
**Notes:** Answer is technically correct but incomplete. Table data on Page 5 was flattened during PDF extraction so the tier details (which SI gets which room category) were lost.

---

### Q3: Is maternity covered? What is the sub-limit?
**Expected:** [Yes. The policy covers maternity (Delivery Expenses), including: Normal delivery, Caesarean section, Pre-natal and post-natal expenses. Sub-limit is limited to up to 10% of the Sum Insured for each delivery.]
**Actual:** ["answer": "Maternity, specifically Delivery Expenses, is covered under this policy. The sub-limit for Delivery Expenses, including pre-natal and postnatal expenses, is up to 10% of the Sum Insured (Page 0, B(13))."]
**Retrieval correct?:** Yes 
**Score:** ✅ 
**Notes:**

---

### Q4: What are the exclusions?
**Expected:** [The main exclusions are: Pre-existing diseases (PED),Specified diseases/procedures – Certain listed diseases and surgeries (e.g., cataract, ENT diseases, thyroid disorders, hernia, gallstones, kidney stones, joint replacement, spinal disorders, gynecological conditions, prostate diseases, hemorrhoids, etc.) are excluded for the first 24 months unless caused by an accident.
30-day waiting period, Investigation and evaluation, Rest cure, rehabilitation, and respite care, Obesity or weight-control treatment,
Cosmetic or plastic surgery unless medically necessary due to accident, burns, or illness, Hazardous or adventure sports-related injuries.
Breach of law, Treatments outside India. treatment for war/nuclear risks, unproven treatments, and certain non-medical expenses]
**Actual:** ["The exclusions are listed on page 1 and include: \nI. Any hospital admission primarily for investigation diagnostic purpose (C(4)), \nII. Pregnancy, infertility (C(17) and C(18)), \nIII. Treatment outside India (D(23)), \nIV. Circumcision, sex change surgery, cosmetic surgery & plastic surgery (C(7), C(8) and C(19)). \nAdditionally, there are other exclusions mentioned on page 1, such as pre-existing diseases (C(1)), specific waiting period (C(2)), and initial waiting period (C(3)), but the full listing is referred to the policy clauses. (Page 1)"]
**Retrieval correct?:** Yes 
**Score:** ✅
**Notes:**

---

### Q5: What is the sum insured options available?
**Expected:** [The Sum Insured options available under the Star Health Assure Insurance Policy are: ₹5 lakh, ₹10 lakh, ₹15 lakh, ₹20 lakh, ₹25 lakh, ₹50 lakh, ₹75 lakh, ₹1 crore, ₹2 crore]
**Actual:** ["I don't have enough information in the provided policy document to answer this question."]
**Retrieval correct?:**  No
**Score:**  ❌
**Notes:** Sum insured options are listed in a table on Page 5 which got 
flattened during extraction. The numbers exist in ChromaDB but without 
table context, retrieval couldn't match "sum insured options" to those chunks. Classic table extraction failure.

---

### Q6: What is the ICU charge limit?
**Expected:** [No separate ICU limit mentioned - system should say "I don't have enough information"
]
**Actual:** ["I don't have enough information in the provided policy document to answer this question."]
**Retrieval correct?:** Yes 
**Score:** ✅ 
**Notes:** Correctly refused to hallucinate. Prompt engineering working as intended.

---

### Q7: Are AYUSH treatments covered?
**Expected:** [Yes. AYUSH treatments are covered under the Star Health Assure Insurance Policy.]
**Actual:** ["Yes, AYUSH treatments are covered. I found this information on page 18 and page 19, and also on page 7, which states that \"AYUSH Hospital is payable up to the sum insured.\" However, it's noted on page 7 that \"Yoga and Naturopathy systems of treatments are excluded from the scope of coverage under AYUSH treatment\"."]
**Retrieval correct?:** Yes 
**Score:** ✅ 
**Notes:**

---

### Q8: What is the free look period?
**Expected:** [The Free Look Period is 15 days from the date you receive the policy document. It applies only to new individual health insurance policies. It does not apply to: Policy renewals, Porting, Migration of policies. During these 15 days, you can review the policy terms and return the policy if you do not find them acceptable.]
**Actual:** ["The free look period is fifteen days from the date of receipt of the policy document, as stated on page 13. This allows the insured person to review the terms and conditions of the policy and return it if not acceptable. Additionally, it's mentioned again on page 13 in section 15 that the Free Look Period shall be applicable on new individual health insurance policies."]
**Retrieval correct?:** Yes 
**Score:** ⚠️ 
**Notes:** Answer is correct in substance but cited wrong page (Page 13 vs  actual Page 14). Minor metadata error — likely a page numbering mismatch between PDF pages and policy document page numbers.

---

### Q9: Is there coverage for day care procedures?
**Expected:** [Yes. The policy covers all day care treatments, meaning medical or surgical procedures that do not require 24-hour hospitalization due to advances in medical technology. These procedures are covered under the inpatient benefits of the policy. Additionally, certain modern treatments are specifically covered whether performed as inpatient or as day care treatment in a hospital, subject to the policy terms and conditions.]
**Actual:** ["Yes, there is coverage for day care procedures. According to page 5, \"All day care treatments are covered.\" Additionally, on page 7, it is mentioned that certain procedures will be covered \"either as in patient or as part of day care treatment in a hospital\"."]
**Retrieval correct?:** Yes 
**Score:** ✅
**Notes:**

---

### Q10: What happens if I miss a premium payment?
**Expected:** [Renewal premium: A 30-day grace period is allowed to renew the policy and retain continuity benefits. However, coverage is not available during the grace period.
Installment premium (Half-yearly/Quarterly): A 7-day grace period is provided. If the premium is not paid within this period, the policy will be cancelled. Coverage is not available from the due date until the premium is received.]
**Actual:** ["If you miss a premium payment, the policy will get cancelled if the instalment premium is not received within the grace period (Page 13). However, no interest will be charged if the instalment premium is not paid on the due date (Page 13). Additionally, if a claim is made, all subsequent premium instalments shall immediately become due and payable (Page 13)."]
**Retrieval correct?:**  Partial
**Score:** ⚠️ 
**Notes:** Correctly mentioned grace period and cancellation but missed the key detail — 30-day grace for renewal vs 7-day grace for instalment premiums. Also didn't mention that coverage is not available during the grace period. Partial retrieval got instalment clause but missed renewal clause.

---

## Summary
- Total: 10 questions
- ✅ Correct: 4 (Q3, Q6, Q7, Q9)
- ⚠️ Partial: 5 (Q1, Q2, Q4, Q8, Q10)
- ❌ Wrong: 1 (Q5)
- Accuracy: 4/10 fully correct, 9/10 directionally correct

## Known Failure Modes Observed
- [x] Table data retrieved without column context (Q2, Q5)
- [ ] Answer confidently wrong (hallucination) — did NOT occur ✅
- [x] Retrieved wrong clause for similar-sounding topics (Q1)
- [x] Missing specific numbers/tiers due to table flattening (Q2, Q5)
- [x] Page number mismatch between PDF and policy numbering (Q8)
- [ ] Slow response (>5 seconds) — note actual latency here

## What to Fix Next
1. Table-aware PDF extraction — biggest impact (fixes Q2, Q5)
2. Better chunking to keep related clauses together (fixes Q1, Q10)
3. Metadata filtering by section to reduce wrong clause retrieval (fixes Q1)