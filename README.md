# CVmatch
Check CV against job position requirements using LLM.

The idea and original code:
https://medium.com/data-science-collective/when-an-ai-tool-i-built-evaluated-my-resume-i-learned-what-100-rejections-never-taught-me-8e8eea1f3d8f

## Usage

```bash
python cvmatch.py resume.pdf job_description.txt
```

### Optional Arguments

- `--eval-prompt`: Path to custom evaluation prompt file (default: prompts/evaluation_prompt.txt)
- `--improve-prompt`: Path to custom improvement prompt file (default: prompts/improve_prompt.txt)

Example with custom prompts:
```bash
python cvmatch.py resume.pdf job_description.txt --eval-prompt my_eval_prompt.txt --improve-prompt my_improve_prompt.txt
```

## Custom Prompts

You can create your own prompt files to customize how the LLM evaluates and improves resumes. The prompt files should include placeholders `{resume}` and `{job_desc}` which will be replaced with the actual resume and job description content.

Example evaluation prompt:
```
Evaluate the following resume against the job description. 
Give a score out of 100, a short rationale, and improvement suggestions if any.

Resume:
{resume}

Job Description:
{job_desc}
```
