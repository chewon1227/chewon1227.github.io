---
layout: default
title: "일반 API와 Batch API 사용하기" 
permalink: /nlp/batchapi/
subtitle: 언제 어떤 상황에서 무슨 방법을 쓰는 것이 효율적일까? 
use_math : true
parent: nlp
---


최근 연구실에서의 작은 성과와 관련해서 정보 아카이빙을 위한 글을 남기려 한다. GPT API를 호출하여 raw data에 prompt를 적용하여 응답을 받아낸 후 정리하는 것이다.

단일 데이터를 넣어서 뽑는 것은 전혀 어렵지 않지만 대량의 데이터를 처리하는 것은 그리 단순한 작업이 아니기 때문에 차이가 있다.

---

## 1. 일반 API 이용하기

전혀 어렵지 않다. 원리는 다음과 같다.

- 구매한 API key를 불러온다. 가져올 모델은 마음대로 (나는 GPT-4를 이용했다)
- JSON 파일을 만들어서 한 줄 한 줄 기존 데이터와 응답 response를 넣어준다 (이때 `time.sleep(n)` 까먹지 말자! )
- 최종 저장하기 (CSV가 가시적으로 보기엔 편해서 엑셀로 저장했다)

이 작업은 생성되는 응답이 그저 숫자 하나씩일 때는 잘 작동하였다. 그러나 API를 매번 호출해야 한다는 문제가 있다. 효율적이지도 않을 뿐더러... 우리는 돈이 많지 않다.

이를 해결하기 위해 Batch API이라는 것을 사용해보았다.

---

## 2. Batch API 이용하기

역시 원리를 이해하면 어렵지 않다.

Batch API는 응답이 바로바로 생성되는 것이 아니라 batch에 올리고, 검증 단계를 거치고, progress되다가, 이후에 complete되기 때문에 즉각적인 응답이 필요한 경우가 아닐 때 사용하기 좋다.

Batch를 이용해서 응답을 하나하나 넣는 게 아니라, batch_size로 설정한 개수만큼 API에 들어가는 것이다. 대신 그만큼 `token_limited_error`에 빠지지 않도록 조심해줘야 한다. `max_token`을 통해서 생성되는 응답의 길이를 조절해줄 수도 있다.

- batch 사이즈와 iterow를 설정해준다
- prompt를 넣되, 역시 JSON 파일을 기본으로 진행한다. JSON 한 줄 한 줄에 prompt를 넣는다.
- batch input file을 업로드한다.
    
    ```python
    batch_input_file = client.files.create(
        file=open("batchinput.jsonl", "rb"),
        purpose="batch"
    )
    ```
    
- batch를 생성한다.
    
    ```python
    batch_input_file_id = batch_input_file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )
    ```
    
- batch의 상태를 확인한다. `validating` → `in progress` → `finalizing` → `completed` 순서대로 가면 잘 하고 있는 거다. `failed`가 나오면 error을 통해 메세지를 확인해야 한다 (보통 `token_limited` 이슈이다)
    
    ```python
    batch = client.batches.retrieve("batch_abc123")
    print(batch)
    ```
    
- batch가 잘 생성되었다면 결과를 가져와준다
    
    ```python
    file_response = client.files.content("file-xyz123")
    print(file_response.text)
    ```
    

사실 장문의 글을 생성하게 된다면, 글 자체를 그대로 응답 데이터로 사용할 수도 있지만 그걸 가공해야 하는 경우가 더 많다.

이래서 파이썬 기초를 배워야 하는구나 싶기도 하다. 너무 귀찮아.

여튼 잘 데이터프레임으로 저장 후 CSV로 구워내면 끝이다.

---

## 3. 언제 쓰기 좋은 거지?

**일반 API vs Batch API 비교**

- **일반 API를 사용할 때**
    - 소량의 데이터(수십~수백 개 수준)를 처리할 때
    - 즉각적인 응답이 필요할 때
    - 응답을 실시간으로 활용해야 할 때 (예: 챗봇, 대화형 시스템)
    - API 호출 비용이 크게 부담되지 않을 때
- **Batch API를 사용할 때**
    - 대량의 데이터를 한 번에 처리해야 할 때 (수천~수백만 개)
    - 실시간 응답이 필요하지 않고 일정 시간이 지나도 괜찮을 때
    - API 호출 비용을 절감하고 싶을 때 (Batch는 일반 API보다 비용 효율성이 높음)
    - 사전 생성된 데이터를 활용해 배치 분석을 할 때

**추가 고려사항**

- 일반 API는 실시간성이 중요할 때 적합하지만, 다수의 요청을 병렬적으로 처리하면 속도 문제가 생길 수 있다.
- Batch API는 대량 처리에 적합하지만 처리 속도가 느리고 검증 단계가 필요하다.
- Token 제한을 잘 고려해서 요청을 설계해야 한다.
- API 비용이 중요한 경우, Batch API를 활용하면 비용을 절감할 수 있다.
- 최적의 방법은 일반 API와 Batch API를 혼합하여 사용하는 것이다. 예를 들어, 실시간으로 일부 요청을 처리하고, 나머지는 Batch API를 통해 백그라운드에서 처리할 수 있다.