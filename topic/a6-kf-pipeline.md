## Kubeflow Pipelines (KFP) 핵심 구성 요소 ##

KFP는 파이썬 코드를 즉시 실행하지 않고, 쿠버네티스에서 실행될 **워크플로우 명세서(YAML/IR)**로 컴파일하기 위해 아래의 핵심 객체들을 사용한다.

### 1. 파이프라인 & 컴포넌트 정의 ###
#### `@dsl.component` (`PythonComponent`) ####
* **역할**: 독립된 컨테이너(Pod)로 실행될 **단일 작업 단위**를 정의합니다.
* **특징**: 일반 파이썬 함수 위에 붙이며, 내부에서 사용할 패키지와 베이스 이미지를 지정할 수 있습니다.

#### `@dsl.pipeline` (`Pipeline`) ####
* **역할**: 여러 컴포넌트를 연결해 **전체 실행 흐름(DAG)**을 조립하는 최상위 함수입니다.
* **특징**: 이 함수 내부에서 컴포넌트를 호출하면 실제 연산이 일어나는 대신 실행 그래프가 구성됩니다.

### 2. 실행 단계 및 데이터 흐름 객체 ###
> * **`PipelineTask` (보통 `task` 변수로 받음)**
>   * **역할**: `@dsl.pipeline` 안에서 컴포넌트를 호출할 때 반환되는 **단계(Step) 객체**입니다.
>   * **활용**: CPU/GPU 자원 할당(`set_memory_limit`), 캐시 제어(`set_caching_options`), 시크릿 주입, 작업 순서 지정(`task_b.after(task_a)`) 등에 사용됩니다.
> * **`Artifact` (`Dataset`, `Model`, `Metrics` 등)**
>   * **역할**: 단계 간에 주고받는 **대용량 데이터나 결과물**을 나타내는 객체입니다.
>   * **활용**: `Input[Dataset]`, `Output[Model]`처럼 타입 힌트로 지정하면 KFP가 S3/MinIO 같은 스토리지에 파일을 자동으로 저장하고 다음 단계로 경로를 넘겨줍니다.
>
> ### 3. 제어 흐름 (Control Flow)
> * **`dsl.If` / `dsl.Elif` / `dsl.Else` (조건 분기)**
>   * **역할**: 이전 단계의 출력값(예: 모델 정확도)에 따라 다음 단계의 실행 여부를 결정합니다. (예: 평가 통과 시에만 배포 실행)
> * **`dsl.ParallelFor` (병렬 반복)**
>   * **역할**: 리스트나 파라미터 목록을 순회하며 여러 태스크를 **동시에 병렬 실행**할 때 사용합니다. (예: 하이퍼파라미터 튜닝)
>
> ### 💡 한눈에 보는 코드 구조
> ```python
> from kfp import dsl
> from kfp.dsl import Input, Output, Dataset, Model
>
> @dsl.component
> def train_model(data: Input[Dataset], model: Output[Model]) -> float:
>     # 실제 학습 로직 및 모델 저장
>     return 0.92
>
> @dsl.pipeline(name="example-pipeline")
> def my_pipeline(raw_data: Input[Dataset]):
>     # 1. 컴포넌트 호출 -> PipelineTask 객체 반환
>     train_task = train_model(data=raw_data)
>     
>     # 2. task 객체로 리소스 및 캐시 설정
>     train_task.set_gpu_limit("1").set_caching_options(False)
>     
>     # 3. 조건 분기 제어
>     with dsl.If(train_task.output > 0.9):
>         deploy_task = deploy_model(model=train_task.outputs["model"])
> ```


> [!info]
>  KFP `task` (`PipelineTask`) 객체란?
> 
> `@dsl.pipeline` 내부에서 컴포넌트(`@dsl.component`)를 호출할 때 반환되는 **파이프라인 단계(Step) 객체**입니다. 함수가 즉시 실행되는 것이 아니라, 쿠버네티스 파드(Pod)로 실행하기 위한 명세서 역할을 합니다.
>
> ### 💡 사용 예시
> ```python
> @dsl.pipeline(name="llm-finetuning-pipeline")
> def my_pipeline(raw_path: str):
>     # 1. 컴포넌트 호출 시 결과값이 아닌 PipelineTask 객체 반환
>     prep_task = preprocess_data(raw_path=raw_path)
>     
>     # 2. cpu() 헬퍼 함수로 자원 및 기본 캐시(cache=True) 설정
>     cpu(prep_task)
>     
>     # 3. 알림·관문·배포 단계는 매번 실행되도록 cache=False로 덮어쓰기
>     alert_task = send_slack_alert(message="학습 완료!")
>     cpu(alert_task, memory="1Gi", cache=False)
> ```
>
> ### 🛠️ `task` 객체로 제어하는 항목
> * **리소스 할당**: `set_cpu_request()`, `set_memory_limit()` 등으로 파드 자원 설정
> * **캐싱 제어**: `set_caching_options()`로 이전 실행 결과 재사용 여부 결정
> * **시크릿 주입**: `kubernetes.use_secret_as_env()`로 API 키 및 환경변수 전달
> * **실행 순서**: `task_b.after(task_a)` 형태로 작업 간 의존성 및 순서 지정
