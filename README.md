# FairHire – AI-Powered Hiring Fairness Analysis

FairHire is an AI-powered hiring analysis application that combines
deterministic data processing with LLM-based agents to evaluate candidate
fit while reducing bias from personally identifiable information (PII).

The application uses **Gemini 3.8 Flash**, **Python**, and **Streamlit** to analyze
resumes against optional job descriptions and produce structured hiring
insights, including skills fit, match scores, and fairness audits.

## Demo

![FairHire Streamlit demo showing a sample resume and its rule-based PII preview](docs/fairhire-demo.jpg)

The screenshot shows the running Streamlit app with a **synthetic** resume. The
rule-based preview replaces the sample email and phone number with `[EMAIL]`
and `[PHONE]`. This preview works without an API key; it is not a Gemini
analysis result.

To try it locally, install dependencies with `pip install -r requirements.txt`
and run `streamlit run app.py`. Paste this sample into **Resume text**:

```text
Alex Taylor
Email: alex.taylor@example.com | Phone: 555-123-4567
Software engineer with 6 years of Python, SQL, and data visualization experience.
Built dashboards and automated reporting pipelines.
```

Optionally enter `Data Analyst: Python, SQL, dashboarding, stakeholder
communication.` as the job description, then expand **Bias-filtered resume
preview (rule-based)**. For the full match-score and fairness analysis, set
`GEMINI_API_KEY` or `GOOGLE_API_KEY` before launching the app and click
**Analyze**. A key with available Gemini API quota is required. Without a key,
**Analyze** reports that a key is required.

## ✨ Key Features

- Upload resumes in **PDF or TXT** format
- Optionally provide a **job description**
- Detect and anonymize sensitive candidate information
- Use an **LLM-based agent** to control the anonymization workflow
- Generate a **0–100 resume–job match score**
- Classify skills as **Strong / Medium / Missing**
- Detect remaining fairness and bias risks
- Produce structured JSON analysis
- Log analysis metadata and results to CSV
- Run consistently through Docker
- Automated CI checks with GitHub Actions

---

## 🏗️ Architecture

The core FairHire workflow is:

```text
Resume + Optional Job Description
              |
              v
       BiasFilterAgent
              |
      +-------+-------+
      |       |       |
    Detect   Mask   Verify
      |       |       |
      +-------+-------+
              |
              v
      Anonymized Resume
              |
              v
     Gemini 3.8 Flash
              |
              v
     Structured Analysis
              |
     +--------+--------+
     |        |        |
 Match     Skills    Bias
 Score      Fit      Audit
              |
              v
         Streamlit UI
              |
              v
          CSV Logging
```

🤖 BiasFilterAgent
A key component of FairHire is the BiasFilterAgent, which uses an
LLM planner together with deterministic Python tools.
Instead of allowing the LLM to directly modify application state or
freely execute operations, the planner selects from a small set of
predefined actions:
detect_pii
mask_pii
verify_and_finish
finish

The Python environment executes the selected action and updates the
agent state.
This separates LLM reasoning from deterministic execution.
Example Agent Loop
State
  ↓
LLM Planner
  ↓
Choose Action
  ↓
Python Tool Execution
  ↓
Update State
  ↓
Repeat / Finish

🛡️ Reliability & Safety Design
Because LLM outputs are probabilistic, FairHire does not rely on the
model for every operation.
1. Deterministic Processing
Predictable operations such as detecting and masking structured
information are implemented using deterministic Python logic and
regular expressions.
The current system masks:
- Email addresses
- Phone numbers
- Years
- Gender-related terms
This makes these operations predictable, inexpensive, and easy to test.
2. Restricted Agent Actions
The LLM planner cannot execute arbitrary operations.
It can only select from:
detect_pii
mask_pii
verify_and_finish
finish

The actual operations are implemented and controlled by Python.
This reduces the chance that an unexpected LLM response directly
changes application behavior.
3. Bounded Agent Execution
The agent loop is limited to a maximum number of iterations to prevent
infinite or repetitive execution.
Repeated PII detection is also detected and automatically redirected
to verification.
4. Structured Output Validation
Both the planner and the final analysis are expected to return
structured JSON.
The application parses and validates these responses before using them.
If the planner returns invalid JSON, the workflow safely terminates
instead of continuing with an unknown action.
5. Fallback Logic
FairHire includes fallback behavior for several failure scenarios.
For example:
- If the Gemini API key is unavailable, deterministic filtering can
  still run.
- If the planner fails to return valid JSON, the agent stops safely.
- If the agent never performs masking, rule-based anonymization is
  automatically applied.
- If the verification step fails, the failure is recorded rather than
  silently trusted.
⚖️ Fairness Pipeline
Before candidate evaluation, FairHire anonymizes the resume.
Original Resume
      ↓
PII Detection
      ↓
Rule-Based Masking
      ↓
LLM Verification
      ↓
Filtered Resume
      ↓
Candidate Evaluation

The final evaluation focuses primarily on the filtered resume so that
skills and experience can be analyzed with less exposure to sensitive
candidate information.
The verifier can also identify residual signals such as:
- Gender
- Age
- School names
- Nationality
- Contact information
and provide additional fairness recommendations.
📊 Structured Analysis
FairHire asks Gemini to return structured JSON containing:
{
  "summary_bullets": [],
  "match_score": 0,
  "skills_fit": {
    "strong": [],
    "medium": [],
    "missing": []
  },
  "bias_flags": [],
  "recommendation": "",
  "bias_filter": {}
}

The Streamlit interface converts this output into an interactive hiring
analysis dashboard.
🖥️ User Interface
The Streamlit application supports:
- Resume upload
- PDF text extraction
- Manual resume input
- Optional job description
- Gemini model selection
- Bias-filtered resume preview
- Resume–JD match score
- Candidate summary
- Skills analysis
- Fairness & bias audit
- Agent execution details
- Recent analysis history
Supported Gemini models:
- gemini-3.8-flash
🧰 Tech Stack
Backend / AI
- Python
- Gemini 3.8 Flash
- Google Generative AI SDK
Application
- Streamlit
Data Processing
- Python Regex
- JSON
- CSV
- PyPDF
Infrastructure
- Docker
- GitHub Actions
📁 Project Structure
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── app.py
├── fairhire_runs.csv
├── requirements.txt
└── README.md

🚀 Running Locally
1. Install dependencies
pip install -r requirements.txt

2. Configure Gemini API Key
Set either:
GEMINI_API_KEY

or:
GOOGLE_API_KEY

3. Start the application
streamlit run app.py

Then open the Streamlit application in your browser.
🐳 Docker
Build the image:
docker build -t fairhire .

Run the container:
docker run fairhire

📈 Analysis Logging
FairHire records analysis metadata in fairhire_runs.csv, including:
- Timestamp
- Model used
- Resume length
- Job description length
- Match score
- Number of bias flags
- Emails removed
- Phone numbers removed
- Years masked
- Gender terms masked
This provides lightweight observability into how the system is being
used.
🔮 Future Improvements
Potential extensions include:
- Stronger PII detection using NER models
- JSON Schema / Pydantic validation
- Automated agent evaluation datasets
- Bias and fairness benchmark suites
- Persistent database storage
- REST API backend
- Prompt and model version tracking
- Production monitoring and observability
- Large-scale data processing pipelines
About
FairHire was developed as a collaborative project exploring how
LLM-based agents and deterministic software can be combined to build
more reliable and bias-aware hiring workflows.
This repository contains a portfolio version of the project focused on
the system architecture, AI agent workflow, reliability mechanisms,
and engineering implementation.
