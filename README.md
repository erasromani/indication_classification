# indication_classification
AI systems developed for breast cancer detection are evaluated at the population level and subgroup level. Subgroup analysis is used to understand the potential limitations of the AI systems and highlight unintended biases present in them. Patient subgroups can be categorized by age, breast tissue density, and clinical indication. While some subgroup categories like age are easy to extract from the electronic health record system, others like clinical indication are only available in unstructured formats and hence difficult to extract. 
In radiology, patients undergo imaging examinations for a variety of reasons (indications), such as screening for a disease or evaluating response to treatment. Knowledge of these indications allows healthcare stakeholders monitor imaging utilization and perform quality assessment. Furthermore, in the era of medical AI systems, many of them are designed, trained and evaluated on examinations performed for a specific indication, such as mammography AI systems for breast cancer screening. Unfortunately, exam indication data is often unavailable in a structured form in electronic health record systems.

In this project, we develop a natural language processing system that classifies breast magnetic resonance imaging (MRI) radiology reports into five clinical indication categories in a true few-shot learning setting \citep{perez}. During a patient breast cancer screening or diagnostic visit, radiology exams are conducted with one or more modalities. Radiologist review the medical images and record their findings in a radiology report. This report should include a description of the reason for the patient’s visit, referred to as the clinical indication. We have categorized the clinical indications into following classes:
\begin{enumerate}
  \item High-risk screening – visit associated with patients that have high risk of cancer, usually because of genetic mutation associated with breast cancer (e.g. BRCA) or history of cancer in patient or patient's family.
  % NOTE: These patients also include intermediate-risk patients, e.g. patients with history of benign biopsies yielding suspicious findings, but not cancer, also some patients with dense breasts.
  % Overall, "high-risk screening" should include patients whose lifetime risk of breast cancer is >15\%.
  \item Pre-operative planning – visit that occurs following a positive biopsy.
  % NOTE: Reasons why these exams are performed: (1) staging, i.e. evaluating the extent of disease and measuring tumors, which is used later to select the most appropriate therapy; (2) detection of breast in contralateral breast (i.e. the other breast, not the one where biopsy was malignant).
  \item Additional workup – visit that occurs after suspicious findings were found on a mammography or ultrasound exam, and the doctor would like to see the suspicious area in a different modality.
  % NOTE: This also includes patients who have symptoms (e.g. nipple discharge, asymmetry) are were referred directly for MRI, without prior mammogram or ultrasound.
  \item Short-term follow-up – visit done shortly (usually within 6 months) after imaging exam or invasive procedure to evaluate stability of findings.
  \item Treatment monitoring – visit done for patients undergoing chemotherapy for breast cancer to monitor how the tumor responds to the treatment.
  % NOTE: Specifically these are patients undergoing *neoadjuvant* chemotherapy (neoadjuvant = before the surgery).
\end{enumerate}
 
