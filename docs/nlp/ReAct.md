---
layout: default
title: "[ReAct] Synergizing Reasoning and Acting in Language Models"
permalink: /nlp/react/
subtitle: 추론과 생각의 시너지를 보는 프롬프팅 기반 ReAct
use_math : true
parent: nlp
---

영어 리딩 공부도 할 겸 한 문장 한 문장 적어보았다. 

<br />


## 1. introduction 

인간 지능의 특이한 특징은, task-oriented action과 verbal reasoning을 원활하게 결합한다는 것이다. 자기 통제, 전략 짜기를 가능하게 하는 인간 인지 및 작업 기억의 유지에 중요한 역할을 한다고 이론화된 것이기도 하다. 부엌에서 요리를 한다고 생각해보자. 두 개의 구체적인 행동 중에, 우리는 과정을 따라가기 위해 (모든 것을 잘랐으므로 물을 담은 냄비를 끓여야겠다), 예외를 조절하거나 상황에 따라서 계획을 수정할 때 (소금이 없으므로, 간장과 후추를 사용하자), 외적인 정보가 필요하다고 느껴졌을 때 (가루 반죽을 어떻게 준비해야 하지? 인터넷에 검색해야겠다) 언어로 생각을 한다. 우리는 또한 (요리책을 열어서 레시피를 읽고, 냉장고를 열고, 재료를 확인한다) 라는 행동을 함 - 생각을 지지하고 질문 (어떤 요리를 지금 당장 할 수 있지?) 에 답하기 위해서. 이렇게 행동(acting)과 추론(reasoning)의 시너지는 보지 못한 상황이 있거나 정보의 불확실성에 마주한다 하더라도 인간들이 새로운 태스크를 빠르게 배우고, 좋은 의사결정과 추론을 할 수 있도록 한다. 

최근 결과들은 자발적인 시스템 (autonomous system)에서 verbal reasoning과 interactive decision making을 합치는 것에 대한 가능성에 힌트를 줘 왔다. 근데, 잘 만들어진 LLM들은 산수, 상식, 상징적인 reasoning task 질문에 대한 답들을 끌어내기 위한 reasoning의 몇 단계를 수행할 수 있는 새로운 가능성을 보여주었다. 그러나, 이러한 chain of thought reasoning은 static black box임 - 모델이 생각을 생성할 때 외적인 세계에 기반을 두고 있지 않고 내적인 상징을 사용하고 있기 때문. 이것은 지식을 업데이트하고 추론을 반응성 있게 하지 못하게 한다. 이것은 fact hallucination이나 error propagation을 만들어낼 수 있다. 반대로, 최근 연구들은 언어 prior을 이용하여 행동을 예측하는 데에 초점을 둬서, 상호작용하는 환경에서 계획하고 행동할 때 사전 훈련된 언어모델을 사용하는 것에 대해 탐구하고 있다. 이러한 접근은 멀티모달 관찰을 텍스트로 바꾸고, 언어모델을 사용해서 domain-specific 행동과 계획들을 만들고, controller을 이용해서 선택하거나 실행하도록 한다. 그러나, 언어 모델이 높은 수준의 목표에 대해 추상적으로 추론하도록 하거나, 작업기억을 유지해서 어떤 행동을 하도록 하지는 않는다. Huang 은 현재 상태의 공간적인 사실에 대해 반복하기 위한 언어 추론의 제한된 형태를 수행하긴 했음. 이러한 몇 안되는 block들과 상호작용하기 위해 간단하게 구현된 태스크들 말고, 어떻게 추론과 행동이 시너지를 내며 합쳐질 수 있는지에 대한 연구가 없다. 그리고, 이런 결합이 추론과 행동을 각각 하는것과 비교해서 시스템적 이득이 있는지에 대한 연구도 없다. 

이 연구에서는 ReAct - 추론과 행동을 언어 모델과 결합하는 패러다임을 제시한다. 이는 다양한 언어 추론과 의사결정 태스크들을 수행한다. ReAct는 LLM에게 태스크와 관련 있는 언어 추론 흔적과 행동을 모두 생성하라는 프롬포트를 제시한다. 이때 교차로 제시하도록 한다 - 모델이 다양한 추론을 생성하고, 유지하고, 행동을 위한 고차원 계획(행동하기 위한 추론)에 맞추도록 교차 배치를 하도록 한다. 그러면서도 추가적인 정보를 추론 과정에 결합하여 외부 환경 (wikipedia) 과 상호작용할 수 있도록 한다. 

우리는 ReAct와 sota 베이스라인에 대해 4가지 벤치마크를 가지고 실증적인 평가를 수행한다 - QA (HotPotQA), 사실 확인(Fever), 텍스트 기반 게임(ALFWorld), 그리고 웹 서치(Webshop). HotPotQA와 Fever에 대해서는, Wikipedia API를 통해 접근 가능하도록 해서, ReAct는 가장 순수한 상태의 생성 모델 (vanilla action) 보다 잘한다 (chain of thought COT 추론 방식과 경쟁적임) . 가장 좋은 접근은 ReAct + CoT 결합하는 것이다. 이러면 내적 지식과 외부적으로 얻어진 정보를 추론할 때 사용할 수 있게 된다. ALFWorld와 Webshop에서는, 2샷 혹은 그냥 1샷 ReAct 프롬프팅이 아주 많은 태스크로 훈련된 모방 / 강화학습보다 잘한다 (각각 거의 34%, 10% 정도 성공 확률이 향상됨). 또한 의사결정에서 sparse하고 변하기 쉬운 추론의 중요성을 주장한다. 일반적인 적용성과 수행 증가를 제외하고, 추론과 행동을 합치는 것은 모델의 해석 능력, 가치, 그리고 진단 능력에 모두 기여한다. 모델의 결정 베이스를 이해할 수 있는 추론 과정을 모두 탐색할 수 있을 뿐만 아니라 인간이 모델의 내적 지식 대 외적 환경에서의 정보를 구별할 수 있다는 점에서 좋다. 

간략하게 말하자면, 우리의 메인 기여도는 다음과 같다 

(1) ReAct를 소개함 - 언어 모델에서 추론과 행동의 시너지를 만들어내는 프롬프트 기반의 새로운 패러다임 

(2) 다양한 벤치마크를 통해 진행한 다양한 실험들 

(3) 시스템적 절제(?)와 분석 - 추론 태스크에서 행동하는 것과 상호작용하는 태스크에서 추론하는 것의 중요성을 이해하기 위함! 

(4) 프롬프팅 셋업에서의 ReAct의 한계를 분석해서 - 추가적인 학습 데이터와 함께 향상할 수 있는 ReAct의 잠재력을 보여주기 위해 초기 파인튜닝 실험을 진행했다. 

강화학습같은 추가적인 패러다임과 결합하거나 더 많은 태스크들에 대해 학습하고 진행하면서 ReAct의 스케일을 점차 키워나가면서 LLM의 잠재력을 더 오픈해나갈 수 있을 것이다. 

<div class="callout">

인간 지능의 핵심은 행동 + 언어적 추론에 있고, 이런 시너지가 인간이 새로운 상황을 빠르게 학습하고 유연한 의사결정을 하게 함. 그치만 LLM들은 행동 / 추론을 따로 진행해옴. 이를 통합하려는 시도들은 추상적인 목표 추론이나 작업 기억 유지 능력이 떨어짐. 

ReAct는 LLM이 언어적 추론과 행동을 교차로 생성하도록 유도함. 언어 추론과 행동을 하나의 프롬프트 내에서 결합해서 상호작용 환경에서 훨씬 더 나은 의사결정이 가능하며, 위키피디아 등 외부 정보를 실시간으로 이용할 수 있음. 

</div>


---
<br />

## 2. REACT : SYNERGIZING REASONING + ACTING 


태스크를 수행하는 환경과 상호작용하는 에이전트의 일반적인 셋업을 생각해보자. 
t 시점에서, 에이전트는 환경에게서 관찰 $o_t$ 를 받고 policy π ( ($a_t|c_t$) ; 이때 $c_t$ 는 에이전트에 대한 맥락 - 즉 특정 맥락일 때 어떤 행동을 하는지에 대해 있는 것이 context) 를 따르는 행동 $a_t$ 를 한다. $c_t$ 에게서 $a_t$ 를 매핑하는 것이 많은 계산을 필요로 하거나 너무 함축적이면 이런 policy를 학습하는 것은 어렵다. 예를 들어, 1c에서 볼 수 있는 에이전트는 QA 태스크를 끝내기 위한 알맞는 최종 행동을 생성하지 못한다. 왜? trajectory context (쭉 따라가는 궤적에 대한 맥락) 에 대한 복잡한 추론을 필요로 하기 때문이다. 비슷하게, 2a 에이전트도 sinkbasin 1이 peppershaker1을 포함하지 않는다는 것을 맥락에서 이해하는 것에 실패함 - 그래서 자꾸 hallucinating 행동을 계속 생성해낸다. 

ReAct는 단순하다 - 에이전트의 action space를 $\hat{\mathcal{A}} = \mathcal{A} \cup \mathcal{L}$ 로 보낸다 ( $\mathcal{A}$ 는 action space, $\mathcal{L}$ 은 language space) (즉, action space와 language space의 겹치는 부분인 새로운 space로 보내는 것). language space에 있는 action인 $aˆt$ - 이는 thought 혹은 reason 과정으로 생각될 것 - 은 외부 환경에 영향을 주지 않으므로, 관찰 피드백도 없다. 대신 생각 $aˆt$ 는 유용한 정보를 구성하기 위해 현 context인 $c_t$ 에 대해서 추론을 진행할거고, 다음 맥락인 $c_(t+1) = (c_t, \hat{a_t})$ 를 업데이트해서 추후 추론과 행동을 돕는다. 매우 다양한 종류의 유용한 생각들이 있다 - 태스크 목표를 쪼개서 행동 계획을 만드는 것, 상식을 이용하는 것, 관찰한 것의 중요한 부분들을 추출해내는 것, 행동 계획을 추적해서 바꾸는 것 등. 

그렇지만 language space $\mathcal{L}$ 은 무한하기 때문에, 증가된 action space에서 학습을 하는 것은 어렵고 강한 language prior이 필요하다. 이 연구에서는, frozen LLM인 PaLM-540B가 도메인 특화된 action을 생성하고 자유 형태의 language thought을 생성하기 위해 **퓨샷 in-context example으로 프롬프팅이 된 상태**를 기본 셋업으로 생각한다. In-context example은 사람이 작성한 action, thought, observation 등의 과정을 포함한다. reasoning이 가장 중요한 태스크에 대해서는 thought, action의 생성을 변화시켜서 태스크를 수행하는 과정이 다양한 thought-action-observation 스텝을 포함할 수 있도록 한다. 반대로, 많은 action을 잠재적으로 포함하고 있는 의사결정 태스크에 대해서는 thought이 그 과정 속에서 가장 관련 있는 위치에만 나타나도록 해서, 언어 모델이 thought와 action의 비동기적 발생을 스스로 결정하도록 한다. 

의사결정과 추론 가능성이 LLM에 통합되어 있기에, ReAct는 몇 가지 특이한 특징을 가지고 있다. 

### **(1) 직관적이고 디자인하기 쉽다** 

ReAct의 프롬프트를 제작하는 것은 아주 직관적이다. 인간 annotator들이 그들이 취한 action에 더불어 자신의 생각을 언어로 입력하면 된다. 이 모델을 위하여 특별히 포맷이 있거나, thought 디자인 등이 사용되지 않았다. 

### **(2) 일반적이고 유연하다.** 

유연한 thought space와 thought-action 발생 형식 덕분에, ReAct는 구별되는 action space와 reasoning need를 가지고 다양한 태스크에 대해 작용한다. 

### **(3) 성능이 뛰어나고 견고하다** 

ReAct는 새로운 태스크에 대해 뛰어난 일반화 능력을 가지고 있다. 1~6개의 in-context 예시들을 통해서만 학습하고, reasoning만 하거나 acting만 하는 다른 베이스라인들보다 일관성 있게 성능이 뛰어나다. 

### **(4) 인간에 맞추어 조정되었고 조절이 가능하다** 

ReAct는 해석 가능한 의사결정 과정과 추론 과정을 보여준다. 인간도 보고 추론과 사실의 정확도를 쉽게 검증할 수 있다. 또한, 인간이 직접 에이전트의 행동을 조절하고 수정할 수 있다. 

<div class="callout">

원래는?

<br />
에이전트가 환경으로부터 $o_t$ 를 받음 → 현재의 맥락 $c_t$ 를 바탕으로 policy $\pi$ 에 따라서 행동 $a_t$ 를 수행함. 

<br />

한계는?

<br />

$c_t$ 가 복잡하거나, $c_t \rightarrow a_t$ 가 너무 어렵다면 제대로 된 행동 $a_t$ 을 뽑아내기가 어려움. 생각을 하지 않고 행동을 하려고 하기 때문. 

<br />

그래서 ReAct는?

<br />

에이전트가 thought도 하도록 만듦. 

<br />

- $\mathcal{A}$ : action space / 실제로 환경을 바꾸는 행동
- $\mathcal{L}$ : language space / 생각이나 reasoning을 말로 표현한 것
- 이 둘을 합쳐서 $\hat{\mathcal{A}}$ 로 표현 ! 즉, 에이전트는 실제 행동을 하거나, thought을 위한 language 행동 둘 중 하나를 선택하게 됨

</div>


---
<br />

## 3. KNOWLEDGE-INTENSIVE REASONING TASKS 

지식 기반 reasoning task에서 시작한다. Wikipedia API와 상호작용하기 때문에 ReAct는 다음에 무엇을 가져올 지 reasoning 하면서 현재 reasoning을 지지할 수 잇는 정보를 취해올 수 있다. (reasoning과 acting의 시너지를 보여주는 것!) 

### 3-1. SETUP 

**Domains** 

지식을 취해오는 것과 reasoning 과정을 위한 두 개의 데이터셋을 고려한다. 

**(1) HotPotQA** : 2개 이상의 Wikipedia 문단들에 대해 reasoning을 진행하는 `multi-hop question` QA 벤치마크

`multi-hop question` : 질문과 함께 거대한 corpus가 주어졌을 때 답을 찾기 위해 다중 추론 점프 (hop)을 수행함. Retriever과 Reader로 이루어져 있음. 한 번의 reasoning으로는 답을 낼 수 없고, A → B → C 처럼 중간 정보를 거쳐서 답에 도달하는 것. 

**(2) FEVER** : 사실 확인하는 벤치마크. 어떤 주장과 Wikipedia 문서가 문맥으로 주어지면, 그 정보를 바탕으로 참/거짓/알 수 없는지를 판단함. ReAct는 직접 Wikipedia 문서를 검색하는 것도 하니까 여기서는 질문만 주어짐. 

**Action Space** 

3개의 action을 포함하고 있는 Wikipedia web API를 만들었다. 

**(1) search[entity]** : 매칭되는 entity가 있을 시 해당하는 wiki page의 첫 5개 문장을 가져온다. 혹은 없다면 Wikipedia 검색 엔진에서 top-5개의 유사한 entity 를 가져온다. 

**(2) lookup[string]** : 페이지에서 string (문자열) 이후에 이어지는 문장을 가져온다. 

**(3) finish[answer]** : 정답을 반환하며 작업을 마친다. 

Action space는 보통 정확한 문서 이름을 기반으로만 문서의 일부분을 가져올 수 있다는 점을 언급한다. 이는 sota 모델들이나 신경 retriever들보다 매우 약하다. 사람들이 어떻게 Wikipedia와 상호작용하는지를 시뮬레이션하고, 모델들이 언어로 명백한 reasoning을 얻어내도록 하는 것이 목적이다. 


### 3-2. METHODS 
<br />

**ReAct Prompting**

HotpotQA와 Fever에 대해서는, training set으로 6개 / 3개 케이스를 선택해서, 프롬프트의 퓨삿 example로서 ReAct-format의 과정을 구성하도록 했다. 각 과정들은 multiple thought-action-observation 스텝을 포함하고 있다. 여기서 free-form thoughts는 다양한 목적으로 이용된다. 구체적으로는, 우리는 thought들의 결합을 질문을 분해하고, Wikipedia에서의 observation으로부터 정보들을 추출하고, 상식을 수행하고, 수학적인 추론을 진행하고, 재해석을 진행하고, 최종 결과를 합성해내는 데에 사용된다. 

**Baselines** 

다양한 베이스라인 구축을 위해서, ReAct 과정을 체계적으로 제거하여 프롬프트를 만든다. 

(1) Standard prompting : ReAct 과정에서 모든 thought, action, observation을 제거한다. 

(2) Chain-of-thought prompting CoT : action, observation을 제거하고 reasoning-only baseline만 둔다. 

(3) Acting-only prompt : thought을 제거하고 WebGPT가 질문에 답하기 위해 인터넷과 어떻게 상호작용하는지를 비슷하게 따라한다. 또한 프롬프팅을 사용하지 않고 모방과 강화학습을 이용한다. 

**Combining Internal and External Knowledge** 

CoT는 reasoning 과정을 만드는 것에는 더 정확하지만 hallucinated 사실과 생각에 쉽게 영향을 받을 수 있는 반면, ReAct의 문제 해결 과정은 좀 더 사실 기반이다. 그렇기에 우리는 ReAct와 `CoT-SC`를 통합하여 모델이 서로 다른 방안으로 언제 바꿀지 결정할 수 있도록 한다. 

`CoT-SC` : CoT는 하나의 추론 경로를 따라가서 정답을 생성하는 것. 반면 CoT-SC는 여러 추론 경로를 따라가서 다양한 정답을 생성한 후 다수결로 최종 정답 생성. 

(1) ReAct → CoT-SC : ReAct가 특정 스텝들에서 정답을 뽑아내지 못한 경우 CoT-SC 방안으로 돌린다. HotpotQA와 FEVER에 대해서 각각 7개 / 5개의 스텝을 설정해뒀다. 더 많은 스텝은 ReAct의 성능을 높이지 못한다. 

(2) Cot-SC → ReAct : n개의 Cot-SC 샘플들에 대해서 majority answer이 n/2 번 이하로 나타날 경우 ReAct로 돌아간다 (과반이 안 넘을 경우!) 

**Finetuning** 

reasoning 과정과 action을 수동적으로 annotating하기에는 어렵기 때문에 bootstraping 접근을 이용한다. ReAct가 생성한 3000개의 정답들과 그 과정들을 이용하여 작은 모델 (PaLM-8/62B) 들을 파인튜닝한다. 이를 통해 모든 thought, action, observation들 (과정들)을 input questions/claims에 조건화될 수 있도록 한다. 


### 3-3. RESULT AND OBSERVATIONS 

**ReAct outperforms Act consistently**

Table 1은 base model을 PaLM-8/62B로 삼아서 다양한 프롬프팅 기법을 적용하였을 때 HotpotQA와 Fever에 대한 성능을 보여준다. ReAct는 두 가지 태스크에 대해 Act보다 잘 한다 - act의 방향성을 제시할 때, 특히 최종 정답을 도출해낼 때 reasoning을 쓰는 가치를 보여준다. 

![Image](https://github.com/user-attachments/assets/8c1dfc5f-ee3e-4dea-b67d-1afa71ebec81)



<br />

**ReAct vs CoT** 

ReAct는 Fever 태스크에서 CoT보다 더 잘하고, HotpotQA에서는 CoT보다 조금 떨어진다. 


A) CoT에서 가장 큰 문제는 hallucination이다. ReAct보다 false positive rate가 훨씬 높고, CoT가 만드는 failure의 가장 큰 부분을 차지한다. 반대로, ReAct는 훨씬 근거 및 사실 기반이다 **(외적 지식 기반에 접근할 수 있기 때문!!)** 

B) reasoning, action, observation을 교차하여 수행하는 것은 ReAct의 groundedness와 trustworthiness를 증가시키지만, 이런 구조적인 억제는 reasoning 스텝에 대한 유연성을 떨어트리기도 한다 - 그래서 CoT보다 Reasoning error을 더 만들기도 한다. ReAct에는 에러를 만드는 패턴이 자주 보이는데, 과거에 했던 thought와 action을 반복적으로 생성하는 것이다. 우리는 다음에 해야 하는 적절한 행동이 무엇인지에 대해 추론하기 실패하였기에 이를 'reasoning error'의 일부로 보기로 했다. 

C) ReAct에게, 성공적으로 지식을 얻기 위해 검색하는 것은 필수적이다. 

**ReAct + CoT-SC perform best for prompting LLMs** 

HotpotQA와 Fever에서 가장 좋은 프롬프팅 방법은 ReAct → CoT-SC 와 CoT-SC → ReAct이다. ReAct + CoT-SC 방법이 각각 하나의 과제에서 강점을 보이지만, 두 방법 모두 샘플 수가 달라져도 CoT-SC보다 일관되게, 그리고 유의미하게 더 좋은 성능을 보인다. 단지 3~5개의 샘플만으로 CoT가 21개의 샘플을 이용했을 때의 성능이 도달한다. 이말인즉슨, reasoning 태스크에서 모델의 내부 지식과 외부 지식을 적절하게 결합하는 것의 중요성을 보여준다. 

**ReAct performs best for fine-tuning** 

PaLM-8/62B에서, ReAct를 프롬프팅하는 것은 4가지 방법 중 가장 좋지 않았다 - in-context 예시들에서 reasoning과 acting을 둘 다 배우는 것이 어려웠기 때문에! 그러나, 3000개의 예시로 파인튜닝하고 나니, ReAct는 가장 좋은 모델이 되었다 ! 그러나 standard나 CoT를 파인튜닝하는 것은 ReAct와 Act를 파인튜닝하는 것보다 성능이 확실히 나빴다. 왜냐면, 전자는 모델이 암기하도록 (잠재적으로 halluincated) 하고, 후자는 모델에게 어떻게 Wikipedia 정보에 접근하는지를 알려주기 때문이다. 

<div class="callout">

ReAct는 항상 Act보다 낫다 

<br />

= 생각하고 행동하는게 그냥 행동하는 것보다 낫다. 

<br />

ReAct와 CoT - CoT는 hallucination이 많고, ReAct는 외부 지식에 기반한 더 사실적인 reasoning을 함 

<br />

ReAct + CoT-SC의 성능이 가장 높다 ! 

<br />

파인튜닝할 때는 ReAct가 가장 좋다. in-context 에서는 어렵지만, ReAct 포맷으로 파인튜닝하면 가장 좋음. 물고기를 잡아주는 게 아니라 물고기를 잡는 방법을 알려준 것.

</div>

---
<br/>

## 4. DECISION MAKING TASKS 

ReAct를 두 개의 언어 기반 상호작용하는 의사결정 태스크에다가 테스트를 해보았다. 

ALFWORLD : 텍스트 기반 게임으로, 에이전트는 6가지 유형의 과제를 수행한다. 에이전트는 템플릿으로 정해진 특정 행동을 취한다. 

WebShop : 온라인 쇼핑 사이트 환경을 구축해둔 벤치마크이다. 

실험 결과 ReAct가 두 태스크 모두에서 Act보다 뛰어난 성능을 보인다. 그러나 여전히 파인튜닝된 모델에 비해서는 성능이 떨어지므로, 프롬프팅만을 이용해서 모델 성능을 높이는 것이 어렵다는 점을 알 수 있다. 

---

## 5. 느낀 점 

확실히 인공지능이 추구하는 방향이 human-like라는 걸 느꼈다. 인간도 태스크를 수행할 때 internal knowledge나 external knowledge 하나에만 의존하거나, action이나 reasoning 하나에만 의존할 경우 좋은 의사결정을 내리지 못하는데 이걸 딱 인공지능으로 구현해낸 것 아닌가 싶다. 

Table2에서 success와 failure의 type은 어떤 기준으로 가져온 건지 궁금해진다. 또한, ReAct를 구축하면서 CoT를 비교로 가져온 이유가 궁금해진다. 단순히 reasoning과 유사한 특성을 가지고 있어서 였을까? RAG와는 왜 비교하지 않았을까 .. 이유를 간단히 생각해보자면 RAG는 모델 구조의 변경이지만 CoT는 프롬프팅의 문제라서 그런가 싶기도 하다.