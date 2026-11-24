
import boto3
import sagemaker

from sagemaker.core.processing import (
    PipelineSession,
    ScriptProcessor,
    ProcessingInput,
    ProcessingOutput
)

from sagemaker.core.shapes.shapes import (
    ProcessingS3Input,
    ProcessingS3Output,
    S3DataSource
)

from sagemaker.mlops.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    PropertyFile
)

from sagemaker.train.model_trainer import ModelTrainer

from sagemaker.core.training.configs import (
    SourceCode,
    Compute,
    InputData
)

from sagemaker.core.model_metrics import ModelMetrics, MetricsSource
from sagemaker.core.workflow import Join

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.condition_step import ConditionStep

from sagemaker.core.workflow.conditions import (
    ConditionGreaterThanOrEqualTo
)

from sagemaker.core.workflow.functions import JsonGet
from sagemaker.mlops.workflow.pipeline import Pipeline


# ------------------------------------------------------------
# STEP 1: AWS CONFIGURATION
# ------------------------------------------------------------

boto_session = boto3.Session()
region = boto_session.region_name

role = (
    "arn:aws:iam::600298656473:role/service-role/"
    "AmazonSageMaker-ExecutionRole-20250410T100059"
)

bucket = "mlops-demo-nehal-600298656473-us-east-1-an"

pipeline_session = PipelineSession(
    boto_session=boto_session,
    default_bucket=bucket
)

image_uri = (
    "683313688378.dkr.ecr.us-east-1.amazonaws.com/"
    "sagemaker-scikit-learn:1.2-1-cpu-py3"
)

pipeline_name = "TitanicMLOpsPipeline"


# ------------------------------------------------------------
# STEP 2: PREPROCESSING
# ------------------------------------------------------------

preprocessor = ScriptProcessor(
    role=role,
    image_uri=image_uri,
    command=["python3"],
    instance_count=1,
    instance_type="ml.m5.large",
    sagemaker_session=pipeline_session,
    base_job_name="titanic-preprocessing"
)

preprocessing_args = preprocessor.run(
    code="scripts/preprocessing.py",

    inputs=[
        ProcessingInput(
            input_name="input",
            s3_input=ProcessingS3Input(
                s3_uri=f"s3://{bucket}/mlops-pipeline-demo/data/titanic.csv",
                s3_data_type="S3Prefix",
                local_path="/opt/ml/processing/input"
            )
        )
    ],

    outputs=[
        ProcessingOutput(
            output_name="train",
            s3_output=ProcessingS3Output(
                s3_uri=f"s3://{bucket}/mlops-pipeline-demo/output/train",
                s3_upload_mode="EndOfJob",
                local_path="/opt/ml/processing/output/train"
            )
        ),

        ProcessingOutput(
            output_name="validation",
            s3_output=ProcessingS3Output(
                s3_uri=f"s3://{bucket}/mlops-pipeline-demo/output/validation",
                s3_upload_mode="EndOfJob",
                local_path="/opt/ml/processing/output/validation"
            )
        ),

        ProcessingOutput(
            output_name="test",
            s3_output=ProcessingS3Output(
                s3_uri=f"s3://{bucket}/mlops-pipeline-demo/output/test",
                s3_upload_mode="EndOfJob",
                local_path="/opt/ml/processing/output/test"
            )
        )
    ],

    wait=False,
    logs=False
)

processing_step = ProcessingStep(
    name="TitanicPreprocessingStep",
    step_args=preprocessing_args
)


# ------------------------------------------------------------
# STEP 3: TRAINING
# ------------------------------------------------------------

sklearn_trainer = ModelTrainer(
    role=role,
    sagemaker_session=pipeline_session,

    training_image=image_uri,

    source_code=SourceCode(
        source_dir="scripts",
        entry_script="train.py"
    ),

    compute=Compute(
        instance_type="ml.m5.large",
        instance_count=1,
        volume_size_in_gb=30
    ),

    output_data_config={
        "s3_output_path": (
            f"s3://{bucket}/mlops-pipeline-demo/training-output"
        )
    },

    stopping_condition={
        "max_runtime_in_seconds": 3600
    }
)

training_inputs = [
    InputData(
        channel_name="train",
        data_source=S3DataSource(
            s3_data_type="S3Prefix",
            s3_uri=(
                processing_step.properties
                .ProcessingOutputConfig.Outputs["train"]
                .S3Output.S3Uri
            )
        )
    ),

    InputData(
        channel_name="validation",
        data_source=S3DataSource(
            s3_data_type="S3Prefix",
            s3_uri=(
                processing_step.properties
                .ProcessingOutputConfig.Outputs["validation"]
                .S3Output.S3Uri
            )
        )
    )
]

training_args = sklearn_trainer.train(
    input_data_config=training_inputs,
    wait=False,
    logs=False
)

training_step = TrainingStep(
    name="TitanicTrainingStep",
    step_args=training_args
)


# ------------------------------------------------------------
# STEP 4: EVALUATION
# ------------------------------------------------------------

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

eval_processor = ScriptProcessor(
    role=role,
    image_uri=image_uri,
    command=["python3"],
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=pipeline_session,
    base_job_name="titanic-evaluation"
)

evaluation_args = eval_processor.run(
    code="scripts/evaluate.py",

    inputs=[
        ProcessingInput(
            input_name="model",
            s3_input=ProcessingS3Input(
                s3_uri=(
                    training_step.properties
                    .ModelArtifacts.S3ModelArtifacts
                ),
                s3_data_type="S3Prefix",
                local_path="/opt/ml/processing/input/model"
            )
        ),

        ProcessingInput(
            input_name="test",
            s3_input=ProcessingS3Input(
                s3_uri=(
                    processing_step.properties
                    .ProcessingOutputConfig.Outputs["test"]
                    .S3Output.S3Uri
                ),
                s3_data_type="S3Prefix",
                local_path="/opt/ml/processing/input/test"
            )
        )
    ],

    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            s3_output=ProcessingS3Output(
                s3_uri=(
                    f"s3://{bucket}/mlops-pipeline-demo/"
                    "output/evaluation"
                ),
                s3_upload_mode="EndOfJob",
                local_path="/opt/ml/processing/evaluation"
            )
        )
    ],

    wait=False,
    logs=False
)

evaluation_step = ProcessingStep(
    name="TitanicEvaluationStep",
    step_args=evaluation_args,
    property_files=[evaluation_report]
)


# ------------------------------------------------------------
# STEP 5: MODEL METRICS
# ------------------------------------------------------------

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=Join(
            on="/",
            values=[
                evaluation_step.properties
                .ProcessingOutputConfig.Outputs["evaluation"]
                .S3Output.S3Uri,
                "evaluation.json"
            ]
        ),
        content_type="application/json"
    )
)


# ------------------------------------------------------------
# STEP 6: MODEL REGISTRATION
# ------------------------------------------------------------

model_builder = ModelBuilder(
    s3_model_data_url=(
        training_step.properties.ModelArtifacts.S3ModelArtifacts
    ),
    image_uri=image_uri,
    role_arn=role,
    sagemaker_session=pipeline_session,

    source_code=SourceCode(
        source_dir="scripts",
        entry_script="inference.py"
    )
)

register_args = model_builder.register(
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="titanic-mlops-group",
    approval_status="PendingManualApproval",
    model_metrics=model_metrics
)

register_step = ModelStep(
    name="TitanicRegisterModel",
    step_args=register_args
)


# ------------------------------------------------------------
# STEP 7: ACCURACY CONDITION
# ------------------------------------------------------------

condition_step = ConditionStep(
    name="TitanicAccuracyCondition",

    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="metrics.accuracy"
            ),
            right=0.75
        )
    ],

    if_steps=[register_step],
    else_steps=[]
)


# ------------------------------------------------------------
# STEP 8: CREATE AND UPDATE PIPELINE
# ------------------------------------------------------------

pipeline = Pipeline(
    name=pipeline_name,

    steps=[
        processing_step,
        training_step,
        evaluation_step,
        condition_step
    ],

    sagemaker_session=pipeline_session
)

pipeline.upsert(role_arn=role)

print("SageMaker Pipeline upserted successfully!")
print("Pipeline name:", pipeline_name)
print("AWS Region:", region)
