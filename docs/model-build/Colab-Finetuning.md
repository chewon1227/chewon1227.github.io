---
layout: post
title: "Colab으로 파인튜닝(Fine-Tuning)하기"
permalink: /build/colab-finetune/
subtitle: Colab, HuggingFace를 이용해서 llama base prompting을 해보자 
parent: model-build
---

이 글은 연세대학교 AIC3110 강의를 참고하였으며, 허가 하에 작성되었습니다. 

프롬프팅에 이어 파인튜닝을 해보려 한다 ! 


---

## 1. 파인튜닝이란?

파인튜닝이란 처음부터 데이터를 쌓아서 학습시키는 것이 아니라, 원래 있던 모델을 목적에 맞게 추가적인 데이터를 통해 학습시켜 특정 태스크를 잘할 수 있는 모델로 성장시키는 과정이다. 

당연히, high-quality 데이터들이 필요할 것이다. high-quality 데이터란 ,, 거의 완전히 real data 즉 인간 데이터이다. 그치만 이런 데이터들은 보안 문제도 있고, 무제한 생성하기도 어렵고, 개인 사생활 문제도 있다. 

그래서 요즘은 **synthetic data** 를 통해 대체하는 추세이다. LLM을 통해 synthetic을 만들어 human-generate data를 대체한다. 

Prompt Chaining을 통해, 큰 모델을 통해 데이터셋을 만들고 그것을 작은 모델에 학습시켜보는 과정을 진행해보려 한다. 


---


## 2. 파인튜닝을 해보자

목적은, 심리상담 챗봇을 만드는 것이다. 
그렇다면 우선 심리 상담 데이터가 필요할 것이다. 


### (1) 심리 상담이 무엇인가 ?

우리가 만들고자 하는 심리 상담 데이터는 현실 상담 시나리오랑 비슷하게 만드는 것이 목표이다. 

그렇다면 현실 상담이란 무엇인가? 

- Multi-turn 으로 진행한다
- CBT(인지행동치료)를 적용한다

사실 모든 치료에 대해 CBT가 근거기반치료인 것은 아니다.. (아직 연구되고 있는 분야가 많음) 그치만 CBT는 거의 모든 임상군에게 적용되었을 때 유의미한 효과를 나타내기 때문에 CBT를 특징으로 잡고 가도 크게 임상적 문제는 없을 것이라고 생각이 된다 

**PatternReframe**이라는 데이터셋을 사용해볼 것이다 (Maddela et al)**.** Persona, Negative Thought, Patters, Reframed Thought 4가지 요소로 구분되어 있다. 이 데이터셋을 이용해서 client를 시뮬레이션해보려 한다. 그 중 하나를 가져왔다. 

```python
user_data = {
        "persona": "i love computers. i'm very good at math and science . i started working at google last week on self driving car research . i i love logical and rational thinking .",
        "thought": "I was rejected by a woman. I am sure it is because she like tough guys and not nerds.",
        "reframes": [
            "I was rejected by a woman but who cares as there's someone for everyone and I can meet others!",
            "I was rejected by a woman. I think I will find someone better soon.",
            "I was rejected by a woman but thats ok ill find a better match soon!"
        ],
        "patterns": [
            "mental filtering",
            "jumping to conclusions: mind reading",
            "overgeneralization",
            "black-and-white or polarized thinking / all or nothing thinking"
        ]
    }
```

persona, negative thought 등의 요소로 이루어져 있다. negative thought → reframes가 CBT의 목표라고 생각하면 된다. pattern의 경우 thought이 포함하고 있는 인지 오류들이다. 



### (2) 심리 상담 데이터를 구조화해보자

심리 상담 데이터를 구조화하기 위해 다음과 같이 세분화하였다. 

- Client-side Simulation : LLM에게 PatternReframe 기반으로 정보를 제공한 뒤, 이를 가지고 psychology Intake Form을 적어보도록 함
- Counselor-side Simulation : Client 상황에 맞는 CBT 기법을 고르고 plan하도록 함. Plan-and-Solve Prompting에서 근거를 찾을 수 있을 것
- Dialogue Generation : Script Mode를 사용할 것
    - Script Mode : 한 모델이 양쪽의 대화를 모두 생성함 (자연스러운 대화가 나옴) .
    - Two-agent Mode : 두 모델이 각각 역할을 맡아 대화를 생성함

> Client side simulation (Form 형성) → CBT 기법 고르기 → 상담 plan → 대화 생성
> 

일반적으로 LLM에게 한 번에 시키면 잘 못하는 것들을, 태스크를 여러 개로 쪼개고 프롬프팅 기법을 통해 더 좋은 품질의 데이터셋을 만들기 위해 노력한 과정이라고 이해하면 좋을 것 같다. 

1. client-side Simulation

우선 클라이언트를 모델링해보자. 이 클라이언트가 왜 상담하러 온 것인지 자세히 적는 것이다. User의 정보를 통해서 클라이언트에 대한 정보나, 왜 상담에 왔으며, 어떤 문제를 가졌는지 등 정보들을 포함해서 적도록 한다. 이런 식으로 example을 준 후, 이 형식에 따라서 적어주도록 했다(one-shot). 모델은 `llama3-70b-8192`를 이용했다. 

```python
# Step 1. Client modeling
persona = user_data['persona']
thought = user_data['thought']
patterns = user_data['patterns']

intake_form_generation_prompt = f'''
Thought depicts a situation where cognitive distortions exhibited by the client have caused problems in daily life, and patterns refer to the types of cognitive distortions the client possesses.

Please generate a client intake form ...

1. Basic Information
- occupation, ...

2. Presenting Problem
- What issue/symptoms ... 

3. Reason for Seeking Counseling
- What was the ... 

4. Past History (including medical history)
- Have you experienced ... 

5. Academic/occupational functioning level (attendance, grades/job performance, etc.)
- Interpersonal ... 

6. Is there anyone you can talk to or get help from when you encounter difficulties or problems?

## Example 1
~~~

## Example 2
[Persona]
{persona}

[Thought]
{thought}

[Patterns]
{patterns}

[Client Intake Form]'''

intake_form = generate_response(
    prompt = intake_form_generation_prompt,
    model_name = "llama3-70b-8192"
)
intake_form
```

내용이 너무 길어서 중략시켰다. 사실 지금 내용 자체가 중요하지는 않다. 

이제 이 인지오류를 겪는 사람에게 어떤 CBT가 적합할지 찾는 과정을 진행해보자. 이를 `cbt_tech_generation_prompt` 로 지정한다. 

```python
# Step 2. CBT Technique Generation
cbt_tech_generation_prompt = f'''
You are a counselor specializing in CBT techniques. Choose top 1 appropriate CBT technique from the given CBT techniques to use with the client based on their intake form. Output only the name of the CBT techniques.

[Types of CBT Techniques]
Efficiency Evaluation, Pie Chart Technique, Alternative Perspective, Decatastrophizing, Pros and Cons Analysis, Evidence-Based Questioning, Reality Testing, Continuum Technique, Changing Rules to Wishes, Behavior Experiment, Problem-Solving Skills Training, Systematic Exposure

## Example 1
[Intake form written by client]
<Reason for Seeking Therapy>
I've been struggling with my temper, ...

[CBT technique]
Alternative Perspective

## Example 2
[Intake form written by client]
{intake_form}

[CBT technique]'''

cbt_tech = generate_response(
    prompt = cbt_tech_generation_prompt,
    model_name = "llama3-70b-8192"
)
cbt_tech
```

실행을 해보면, `Evidence-Based Questioning` 이라는 결과가 나온다. 즉, 이 상황에 대해 적용할 수 있는 CBT 기법이 바로 Evidence-Based Questioning이라는 것이다. llama 모델은 사전학습된 데이터가 있기에 이를 활용하여 입력된 텍스트를 분석하고, 가장 적절한 것을 선택한 것이다. 

2. counselor-side Simulation

이제 구체적인 플랜을 짜보자. 

```python
# Step 3. CBT Planning

cbt_planning_prompt = f'''
You are a counselor specializing in CBT techniques. Plan to counsel the patient who has completed ... 

## Example 1
[Intake form written by client]
<Reason for Seeking Therapy>
I've been struggling with ... 

<Goals for Therapy>
I want to find ways to stay ... 

<Cognitive Distortions Observed>
All-or-nothing thinking: The client ... 

[CBT technique]
Decatastrophizing

[Counseling plan]
Decatastrophizing
1. Identify Catastrophic ... 

{intake_form}

[CBT technique]
{cbt_tech}

[Counseling sequence]
'''

cbt_plan = generate_response(
    prompt = cbt_planning_prompt,
    model_name = "llama3-70b-8192"
)
cbt_plan
```

아주 좋은 결과를 내준다. 

3. dialogue 제작 

이를 가지고 dialogue를 만들어 볼 것이다 

```python
# Step 4. Dialogue generation

dialogue_generation_prompt = f'''
Your task is to generate a multi-turn counseling dialogue between a client and a professional counselor. Generate a dialogue that incorporates the following guidelines:

# General guidelines
1. The dialogue is ...

# Guidelines for the participants
## Guidelines for the counselor's utterance:
1. At the start of the conversation, ...

## Guidelines for the client's utterance:
1. Engage authentically with the counselor's inquiries ... 

[Situation of the client]
{intake_form}

[Counseling plan]
{cbt_plan}

Remember that you are an independent dialogue writer and should finish the dialogue by yourself.

[Generated dialogue]
'''

dialogue = generate_response(
    prompt = dialogue_generation_prompt,
    model_name = "llama3-70b-8192"
)
dialogue
```

그러면 멀티턴 대화를 만들어준다. 

보면 알겠지만, 지난 번에 만들었던 generate_response 함수에 프롬포트와 모델명을 넣어서 그냥 계속 생성하는 방식이다. 예시를 한개씩 주고 있으니 one-shot이 되겠다. 어렵지 않은 과정이지만, 프롬포트를 만들 때 임상/상담 전문가가 필요하겠다. 



### (3) 파인튜닝 전 기본 작업들

모델은 LLaMa2-7b , 데이터셋은 Cactus를 사용한다 (https://huggingface.co/datasets/DLI-Lab/cactus). Cactus는 위의 과정을 계속 반복하여 생성된 데이터셋이다. 

우선 필요한 자료들을 설치해준다. 

```python
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q trl xformers wandb datasets einops gradio sentencepiece bitsandbytes

from datasets import load_dataset
ds = load_dataset("DLI-Lab/cactus",split='train')
```

파인튜닝을 할 때 기본적으로 GPU가 많이 들기 때문에, 딱 10개 데이터만 가지고 학습을 진행해볼 것이다. `train_dataset` 한 개 한 개를 보면 아까 돌았던 사이클 (client-side, counselor-side로 만든 multi-turn dialogue) 이 잘 저장되어 있다. 

지금까지의 dialogue history가 주어졌을 때, 그걸 바탕으로 counselor이 적합한 대답을 하는 걸 학습하는 방식을 통해 상담 챗봇을 만들어본다. 

```python
train_dataset = ds[:10]['dialogue']
refined_dataset = []
for dialogue in train_dataset:
  splited_dialogue = dialogue.split('\n')
  for i in range(len(splited_dialogue)//2-1):
    data = {}
    data['dialogue_history'] = ''.join(splited_dialogue[:i*2+2])
    data['response'] = splited_dialogue[i*2+2]
    refined_dataset.append(data)
refined_dataset[0]
```

refined_dataset을 보면 이제 'dialogue_history'와 'response'로 나뉘어 딕셔너리 형태로 저장되는 것을 볼 수 있다. 이런 식으로 .. 

```
{'dialogue_history': "Counselor: Good afternoon, Brooke. Thank you for joining me today. Can you tell me a bit about what brings you to counseling?Client: Hi. I've been really anxious about going back to the animal shelter where I volunteer. I feel like the animals will hate me because they didn't remember me the last time I visited. It's been really tough.",
 'response': "Counselor: I'm sorry to hear that you've been feeling this way. It sounds like this is something that's been troubling you for a while. Can you tell me more about what happened during your last visit to the shelter?"}
```

```python
def make_data_module(x):
  System_prompt = f"You are playing the role of a counselor in a psychological counseling session. Your task is to generate the next counselor utterance in the dialogue. The goal is to create a natural and engaging response that builds on the previous conversation."
  User_prompt = f"Counseling Dialogue History:\n{x['dialogue_history']}"
  output_text = f"{x['response']}"
  input_text = f"<|start_header_id|>system<|end_header_id|>\n\n{System_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{User_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
  return {"input": input_text, "output": output_text}

transformed_dataset = [make_data_module(item) for item in refined_dataset]
```



### (4) 파인튜닝 진짜 해보자

이제 진짜로 파인튜닝을 해보자. 먼저 hugging face에서 api를 받아서 오자. 

```python
from huggingface_hub import notebook_login
notebook_login()
```

실행하면 나오는 칸에 LLAMA api key를 넣어주면 된다. 

모델은 `Llama-2-7b-hf` 를 사용해보겠다 (다른 거 써도 된다.) 

LLaMA-2 모델을 4비츠 양자화와 LoRA를 사용해서 훈련하는 방식이다. (보통 LoRA는 미세조정 시 많이 사용하고 적은 리소스로 파라미터를 효율적으로 훈련할 수 있도록 한다) 

```python
model_name = "meta-llama/Llama-2-7b-hf"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.float16,
    bnb_4bit_use_double_quant= False,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0}
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token
```

`bnb_config` 는 4비트 양자화를 통해 모델의 메모리 사용량을 줄이고, nf4를 통해 양자화 유형을 정해주었으며 torch.float16으로 계산 데이터 유형을 지정하였다. 

`AutoModelForCausalLM.from_pretrained` 를 통해 사전 학습된 언어 모델을 로드했고, `quantization_config=bnb_config` 를 통해 아까 설정해둔 4비트 양자화를 적용했다. `device_map={"": 0}` 을 통해 모델을 GPU에 로드하도록 했다. 

`add_eos_token`**,** `add_bos_token` 을 통해 문장의 시작과 끝을 자동으로 추가하도록 했다. 

LoRA 설정을 위해 세부 매개변수를 지정하고, TrainingArguments 클래스로 모델 훈련을 위한 하이퍼파라미터를 설정한다. 그리고 사전 학습된 모델을 LoRA 기반으로 미세 조정하는 설정을 구현한다. 

```python
from transformers import TrainingArguments 

peft_config = LoraConfig(
    lora_alpha= 8,
    lora_dropout= 0.1,
    r= 16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj"]
)

training_arguments = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 1,
    per_device_train_batch_size= 1, #기본값: 8 (코랩에서 돌아가도록 하려면 1이 좋다) 
    gradient_accumulation_steps= 2,
    optim = "paged_adamw_8bit",
    save_steps= 1000,
    logging_steps= 30,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    fp16= False,
    bf16= False,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio= 0.3,
    group_by_length= True,
    lr_scheduler_type= "linear",
    report_to=["none"]
)

from datasets import Dataset
from trl import SFTTrainer

hf_dataset = Dataset.from_dict({
    "text": [f"{item['input']}{item['output']}" for item in transformed_dataset]
})
trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
)
```

모든 설정을 끝냈으니 이제 훈련을 해볼것이다. 우리의 모든 설정을 포함하고 있는 `trainer` 을 이용한다. 

```python
trainer.train()
```

데이터 양이 적어서 한 20분이면 끝난다.