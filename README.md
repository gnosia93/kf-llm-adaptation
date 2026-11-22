_This workshop is under construction, Not fully implemented yet ..._

## Building a Domain-Specific LLM : CPT, SFT, DPO & Kubeflow Pipelines ##

`kf-llm-pipeline` is an end-to-end hands-on workshop for building a Kubernetes domain-specific LLM and automating the workflow with Kubeflow Pipelines (KFP). You will walk through the complete model adaptation and post-training lifecycle—from data preprocessing and Continued Pre-Training (CPT) to Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO)—and package the entire process into a reproducible MLOps pipeline.


![](https://github.com/gnosia93/llm-train-ft/blob/main/images/llm-pipeline.jpg)

* C1. 환경설정

* [C2. 데이터 전처리](https://github.com/gnosia93/llm-train-ft/blob/main/topic/2-data-preprocessing.md)

* [C3. 1단계 - Continued Pre-Training (쿠버네티스 배우기)](https://github.com/gnosia93/llm-train-ft/blob/main/topic/3-cpt.md)

* [C4. 2단계 - Supervised Fine-Tuning (대화 스킬 배우기)](https://github.com/gnosia93/llm-train-ft/blob/main/topic/4-sft.md)

* [C5. 3단계 - DPO (선호도 정렬)](https://github.com/gnosia93/llm-pipeline/blob/main/topic/5-dpo-preference.md)

* [C6. Pipeline](https://github.com/gnosia93/llm-pipeline/blob/main/topic/6-pipeline.md)


### Appendix ###

* [LoRA (Low-Rank Adaptation) 핵심 원리](https://github.com/gnosia93/llm-train-ft/blob/main/topic/a1-lora.md)
* [LR Schedule & Weight Decay](https://github.com/gnosia93/llm-train-ft/blob/main/topic/a2-lr-schedule.md)
* [딥러닝 트러블슈팅 가이드](https://github.com/gnosia93/llm-train-ft/blob/main/topic/a3-top-20.md)
* [Reward Model(보상/채점 모델) 만들기](https://github.com/gnosia93/llm-train-ft/blob/main/topic/a4-reward-model.md) 
* [Model FLOPs Utilization](https://github.com/gnosia93/llm-pipeline/blob/main/topic/a5-mfu-calc.md)
* [kfp 구성요소](https://github.com/gnosia93/llm-pipeline/blob/main/topic/a6-kf-pipeline.md)
* [A100 vs H100](https://github.com/gnosia93/kf-llm-pipeline/blob/main/topic/a7-a100_vs_h100.md) / [EC2 사양 비교](https://github.com/gnosia93/vlm-distillation/blob/main/ec2-price.md)
