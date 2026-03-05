import asyncio
import base64
import httpx
import sys
import time
import timeit
import traceback
import typing
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request, Body, Depends, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from openai import AsyncOpenAI

from app.config import config as api_config, LOGGER

sys.path.insert(0, str(Path(__file__).parent.parent))

print("DEBUG: Importing agents...")
try:
    from agents.planner_agent import PlannerAgent
    print("DEBUG: Imported PlannerAgent")
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    print("DEBUG: Imported all agents")
    from utils import config
    from utils.eval_toolkits import get_score_for_image_referenced
    from utils.paperviz_processor import PaperVizProcessor
    print("DEBUG: Imported utils")

except ImportError as e:
    print(f"DEBUG: ImportError: {e}")
    import traceback
    traceback.print_exc()
    raise e
except Exception as e:
    print(f"DEBUG: Exception during import: {e}")
    import traceback
    traceback.print_exc()
    raise e

EXAMPLE_METHOD = r"""## Methodology: The PaperVizAgent Framework

In this section, we present the architecture of PaperVizAgent, a reference-driven agentic framework for automated academic illustration. As illustrated in Figure \ref{fig:methodology_diagram}, PaperVizAgent orchestrates a collaborative team of five specialized agents—Retriever, Planner, Stylist, Visualizer, and Critic—to transform raw scientific content into publication-quality diagrams and plots. (See Appendix \ref{app_sec:agent_prompts} for prompts)

### Retriever Agent

Given the source context $S$ and the communicative intent $C$, the Retriever Agent identifies $N$ most relevant examples $\mathcal{E} = \{E_n\}_{n=1}^{N} \subset \mathcal{R}$ from the fixed reference set $\mathcal{R}$ to guide the downstream agents. As defined in Section \ref{sec:task_formulation}, each example $E_i \in \mathcal{R}$ is a triplet $(S_i, C_i, I_i)$.
To leverage the reasoning capabilities of VLMs, we adopt a generative retrieval approach where the VLM performs selection over candidate metadata:
$$
\mathcal{E} = \text{VLM}_{\text{Ret}} \left( S, C, \{ (S_i, C_i) \}_{E_i \in \mathcal{R}} \right)
$$
Specifically, the VLM is instructed to rank candidates by matching both research domain (e.g., Agent & Reasoning) and diagram type (e.g., pipeline, architecture), with visual structure being prioritized over topic similarity. By explicitly reasoned selection of reference illustrations $I_i$ whose corresponding contexts $(S_i, C_i)$ best match the current requirements, the Retriever provides a concrete foundation for both structural logic and visual style.

### Planner Agent

The Planner Agent serves as the cognitive core of the system. It takes the source context $S$, communicative intent $C$, and retrieved examples $\mathcal{E}$ as inputs. By performing in-context learning from the demonstrations in $\mathcal{E}$, the Planner translates the unstructured or structured data in $S$ into a comprehensive and detailed textual description $P$ of the target illustration:
$$
P = \text{VLM}_{\text{plan}}(S, C, \{ (S_i, C_i, I_i) \}_{E_i \in \mathcal{E}})
$$

### Stylist Agent

To ensure the output adheres to the aesthetic standards of modern academic manuscripts, the Stylist Agent acts as a design consultant.
A primary challenge lies in defining a comprehensive “academic style,” as manual definitions are often incomplete.
To address this, the Stylist traverses the entire reference collection $\mathcal{R}$ to automatically synthesize an *Aesthetic Guideline* $\mathcal{G}$ covering key dimensions such as color palette, shapes and containers, lines and arrows, layout and composition, and typography and icons (see Appendix \ref{app_sec:auto_summarized_style_guide} for the summarized guideline and implementation details). Armed with this guideline, the Stylist refines each initial description $P$ into a stylistically optimized version $P^*$:
$$
P^* = \text{VLM}_{\text{style}}(P, \mathcal{G})
$$
This ensures that the final illustration is not only accurate but also visually professional.

### Visualizer Agent

After receiving the stylistically optimized description $P^*$, the Visualizer Agent collaborates with the Critic Agent to render academic illustrations and iteratively refine their quality. The Visualizer Agent leverages an image generation model to transform textual descriptions into visual output. In each iteration $t$, given a description $P_t$, the Visualizer generates:
$$
I_t = \text{Image-Gen}(P_t)
$$
where the initial description $P_0$ is set to $P^*$.

### Critic Agent

The Critic Agent forms a closed-loop refinement mechanism with the Visualizer by closely examining the generated image $I_t$ and providing refined description $P_{t+1}$ to the Visualizer. Upon receiving the generated image $I_t$ at iteration $t$, the Critic inspects it against the original source context $(S, C)$ to identify factual misalignments, visual glitches, or areas for improvement. It then provides targeted feedback and produces a refined description $P_{t+1}$ that addresses the identified issues:
$$
P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)
$$
This revised description is then fed back to the Visualizer for regeneration. The Visualizer-Critic loop iterates for $T=3$ rounds, with the final output being $I = I_T$. This iterative refinement process ensures that the final illustration meets the high standards required for academic dissemination.

### Extension to Statistical Plots

The framework extends to statistical plots by adjusting the Visualizer and Critic agents. For numerical precision, the Visualizer converts the description $P_t$ into executable Python Matplotlib code: $I_t = \text{VLM}_{\text{code}}(P_t)$. The Critic evaluates the rendered plot and generates a refined description $P_{t+1}$ addressing inaccuracies or imperfections: $P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)$. The same $T=3$ round iterative refinement process applies. While we prioritize this code-based approach for accuracy, we also explore direct image generation in Section \ref{sec:discussion}. See Appendix \ref{app_sec:plot_agent_prompt} for adjusted prompts."""

EXAMPLE_CAPTION = "Figure 1: Overview of our PaperVizAgent framework. Given the source context and communicative intent, we first apply a Linear Planning Phase to retrieve relevant reference examples and synthesize a stylistically optimized description. We then use an Iterative Refinement Loop (consisting of Visualizer and Critic agents) to transform the description into visual output and conduct multi-round refinements to produce the final academic illustration."

EXAMPLE_INPUT_DATA = {
      "Department": [
        "HR",
        "Finance",
        "Sales",
        "IT",
        "Operations"
      ],
      "Revenue": [
        15.3,
        29.2,
        35.8,
        46.0,
        55.7
      ],
      "Profit": [
        52.0,
        41.2,
        32.3,
        27.1,
        38.1
      ],
      "Growth Rate": [
        40.9,
        20.6,
        23.9,
        12.5,
        22.6
      ],
      "Customer Satisfaction": [
        37.0,
        17.8,
        26.2,
        14.1,
        34.5
      ],
      "Market Share": [
        23.4,
        31.3,
        45.5,
        28.1,
        48.6
      ],
      "R&D Spend": [
        86.3,
        73.9,
        54.3,
        49.2,
        39.5
      ]
    }

EXAMPLE_PLOT_CAPTION = "A balloon plot heatmap about Departments and KPIs, titled Business Performance Metrics by Department(size of the desired plot: width=12.0, height=9.0)"

app = FastAPI(
    title="PaperBanana API",
    description="API service for PaperBanana, a system that generates visualizations for scientific papers using large language models and multi-agent collaboration.",
    version="1.0.0"
)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    token = credentials.credentials
    async with httpx.AsyncClient() as http_client:
        try:
            response = await http_client.get(
                url="https://openrouter.ai/api/v1/key",
                headers={
                    "Authorization": f"Bearer {token}"
                },
                timeout=5.0
            )
            response.raise_for_status()
            return token
        except httpx.HTTPStatusError as e:
            LOGGER.warning(f"Token verification failed with status code {e.response.status_code}: {e.response.text}")
            raise HTTPException(status_code=401, detail="Invalid API token")
        except Exception as e:
            LOGGER.warning(f"Token verification failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid API token")

@app.middleware("http")
async def timing_request(request: Request, call_next):

    start = timeit.default_timer()
    request.state.is_disconnected = request.is_disconnected
    response = await call_next(request)
    response.headers["X-Process-Time"] = f'{timeit.default_timer() - start:.6f}'

    return response

@app.exception_handler(Exception)
async def value_error_handler(request: Request, exc: Exception):
    LOGGER.exception("Unhandled exception occurred")
    return JSONResponse({
        'error': str(exc),
        'traceback': "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        'status_code': 500
    }, status_code=500)


async def generate(
    data: dict,
    task_name: str,
    model_name: str,
    image_model_name: str,
    temperature: float,
    max_critic_rounds: int,
    exp_mode: str,
    return_detailed: bool,
    auth_token: str
):
    
    exp_config = config.ExpConfig(
        dataset_name="PaperBananaBench",
        task_name=task_name,
        temperature=temperature,
        exp_mode=exp_mode,
        max_critic_rounds=max_critic_rounds,
        model_name="openrouter-" + model_name,
        image_model_name="openrouter-" + image_model_name,
    )

    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )

    LOGGER.info(f"Processing {task_name} with model_name: {model_name}, image_model_name: {image_model_name}, temperature: {temperature}, max_critic_rounds: {max_critic_rounds}, exp_mode: {exp_mode}")
    with processor.with_config(api_key=auth_token):
        result = await processor.process_single_query(data, do_eval=False)

    return JSONResponse(result) if return_detailed else JSONResponse({f"target_{task_name}_base64_jpg": result[result["eval_image_field"]]})


@app.post('/diagram')
async def generate_diagram(
    request: Request,
    method_section: str = Body(..., description="The method section of the scientific paper to visualize, provided as plain text. Markdown format is recommended.", examples=[EXAMPLE_METHOD]),
    figure_caption: str = Body(..., description="The caption of the figure to generate, provided as plain text. Markdown format is recommended.", examples=[EXAMPLE_CAPTION]),
    model_name: str = Body('google/gemini-3-pro-preview', description="The name of the language model to use for processing."),
    image_model_name: str = Body('google/gemini-3-pro-image-preview', description="The name of the image generation model to use for processing."),
    temperature: float = Body(1.0, description="The temperature setting for the language model, controlling the randomness of the output. Higher values (e.g., 1.0) produce more random outputs, while lower values (e.g., 0.2) produce more focused and deterministic outputs.", gt=0.0, lt=2.0),
    aspect_ratio: typing.Literal['1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '9:21', '21:9'] = Body('21:9', description="The desired aspect ratio for the generated diagram."),
    max_critic_rounds: int = Body(3, description="The maximum number of critique and revision rounds to perform. This controls how many times the agents will iteratively improve the diagram based on feedback.", gt=0),
    pipeline_type: typing.Literal['vanilla', 'planner', 'planner_stylist', 'planner_critic', 'full'] = Body('full', description="The type of pipeline to use for processing. Supported values are 'vanilla', 'planner', 'planner_stylist', 'planner_critic', and 'full'."),
    return_detailed: bool = Body(False, description="Whether to return detailed intermediate outputs from all agents in the response."),
    timeout: typing.Optional[float] = Body(None, description="Optional timeout in seconds for the entire processing of the request. If the processing time exceeds this limit, the task will be cancelled and a timeout error will be returned.", examples=[None]),
    auth_token: str = Depends(verify_token)
):
    data = {
        "content": method_section,
        "visual_intent": figure_caption,
        "additional_info": {
            "rounded_ratio": aspect_ratio,
        },
        "max_critic_rounds": max_critic_rounds,
    }

    task = asyncio.create_task(generate(
        data=data,
        task_name="diagram",
        model_name=model_name,
        image_model_name=image_model_name,
        temperature=temperature,
        max_critic_rounds=max_critic_rounds,
        exp_mode=f"dev_{pipeline_type}" if pipeline_type != "vanilla" else "vanilla",
        return_detailed=return_detailed,
        auth_token=auth_token,
    ))

    counter = 0
    while not task.done():
        if (await request.state.is_disconnected()):
            LOGGER.info("Client disconnected, cancelling the task...")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                LOGGER.info("Task cancelled successfully.")
            return
        await asyncio.sleep(0.1)
        counter += 0.1
        if timeout is not None and counter >= timeout:
            LOGGER.info(f"Request processing exceeded timeout of {timeout} seconds, cancelling the task...")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                LOGGER.info("Task cancelled successfully due to timeout.")
            return JSONResponse({"error": "Request processing exceeded timeout limit."}, status_code=504)

    return await task


@app.post('/plot')
async def generate_plot(
    request: Request,
    input_data: dict = Body(..., description="The input data for generating the plot.", examples=[EXAMPLE_INPUT_DATA]),
    figure_caption: str = Body(..., description="The caption of the figure to generate, provided as plain text. Markdown format is recommended.", examples=[EXAMPLE_PLOT_CAPTION]),
    model_name: str = Body('google/gemini-3-pro-preview', description="The name of the language model to use for processing."),
    image_model_name: str = Body('google/gemini-3-pro-image-preview', description="The name of the image generation model to use for processing."),
    temperature: float = Body(1.0, description="The temperature setting for the language model, controlling the randomness of the output. Higher values (e.g., 1.0) produce more random outputs, while lower values (e.g., 0.2) produce more focused and deterministic outputs.", gt=0.0, lt=2.0),
    aspect_ratio: typing.Literal['1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '9:21', '21:9'] = Body('4:3', description="The desired aspect ratio for the generated diagram."),
    max_critic_rounds: int = Body(3, description="The maximum number of critique and revision rounds to perform. This controls how many times the agents will iteratively improve the diagram based on feedback.", gt=0),
    pipeline_type: typing.Literal['vanilla', 'planner', 'planner_stylist', 'planner_critic', 'full'] = Body('full', description="The type of pipeline to use for processing. Supported values are 'vanilla', 'planner', 'planner_stylist', 'planner_critic', and 'full'."),
    return_detailed: bool = Body(False, description="Whether to return detailed intermediate outputs from all agents in the response."),
    timeout: typing.Optional[float] = Body(None, description="Optional timeout in seconds for the entire processing of the request. If the processing time exceeds this limit, the task will be cancelled and a timeout error will be returned.", examples=[None]),
    auth_token: str = Depends(verify_token)
):
    
    data = {
        "content": input_data,
        "visual_intent": figure_caption,
        "additional_info": {
            "rounded_ratio": aspect_ratio,
        },
        "max_critic_rounds": max_critic_rounds,
    }

    task = asyncio.create_task(generate(
        data=data,
        task_name="plot",
        model_name=model_name,
        image_model_name=image_model_name,
        temperature=temperature,
        max_critic_rounds=max_critic_rounds,
        exp_mode=f"dev_{pipeline_type}" if pipeline_type != "vanilla" else "vanilla",
        return_detailed=return_detailed,
        auth_token=auth_token,
    ))

    counter = 0
    while not task.done():
        if (await request.state.is_disconnected()):
            LOGGER.info("Client disconnected, cancelling the task...")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                LOGGER.info("Task cancelled successfully.")
            return
        await asyncio.sleep(0.1)
        counter += 0.1
        if timeout is not None and counter >= timeout:
            LOGGER.info(f"Request processing exceeded timeout of {timeout} seconds, cancelling the task...")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                LOGGER.info("Task cancelled successfully due to timeout.")
            return JSONResponse({"error": "Request processing exceeded timeout limit."}, status_code=504)

    return await task


@app.post('/polish')
async def polish_image(
    request: Request,
    image_base64: str = Body(..., description="The input diagram or plot to be polished, provided as a base64-encoded string of the image in JPG format."),
    task_name: typing.Literal['diagram', 'plot'] = Body(..., description="The type of the task, either 'diagram' or 'plot'. This helps the polish agent understand the context and apply appropriate polishing strategies."),
    input_data: typing.Optional[dict] = Body(None, description="The original input data used for generating the plot, provided as a JSON object. This is optional and only for plot polishing, as it can help the agent better understand the content and provide more accurate suggestions.", examples=[None]),
    model_name: str = Body('google/gemini-3-pro-preview', description="The name of the language model to use for processing."),
    image_model_name: str = Body('google/gemini-3-pro-image-preview', description="The name of the image generation model to use for processing."),
    temperature: float = Body(1.0, description="The temperature setting for the language model, controlling the randomness of the output. Higher values (e.g., 1.0) produce more random outputs, while lower values (e.g., 0.2) produce more focused and deterministic outputs.", gt=0.0, lt=2.0),
    aspect_ratio: typing.Literal['1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '9:21', '21:9'] = Body('21:9', description="The desired aspect ratio for the generated diagram."),
    return_detailed: bool = Body(False, description="Whether to return detailed intermediate outputs from all agents in the response."),
    timeout: typing.Optional[float] = Body(None, description="Optional timeout in seconds for the entire processing of the request. If the processing time exceeds this limit, the task will be cancelled and a timeout error will be returned.", examples=[None]),
    auth_token: str = Depends(verify_token)
):
    
    exp_config = config.ExpConfig(
        dataset_name="PaperBananaBench",
        task_name=task_name,
        temperature=temperature,
        exp_mode=f"dev_polish",
        model_name="openrouter-" + model_name,
        image_model_name="openrouter-" + image_model_name,
    )

    image_path = Path(exp_config.work_dir) / "data" / "PaperBananaBench" / task_name / f"temp_{auth_token.split('-')[-1]}_{int(time.time())}_{uuid4().hex}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    with open(image_path, "wb") as f:
        f.write(base64.b64decode(image_base64.split(",")[-1]))
    
    data = {
        "path_to_gt_image": image_path.name,
        "additional_info": {
            "rounded_ratio": aspect_ratio,
        },
    }
    if task_name == "plot" and input_data is not None:
        data["content"] = input_data

    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )

    LOGGER.info(f"Processing polish {task_name} with model_name: {model_name}, image_model_name: {image_model_name}, temperature: {temperature}, aspect_ratio: {aspect_ratio}")
    with processor.with_config(api_key=auth_token):
        task = asyncio.create_task(processor.process_single_query(data, do_eval=False))
        counter = 0
        while not task.done():
            if (await request.state.is_disconnected()):
                LOGGER.info("Client disconnected, cancelling the polish task...")
                task.cancel()
                try:
                    image_path.unlink()
                except Exception as e:
                    LOGGER.error(f"Failed to delete temporary image {image_path} after cancellation: {e}")
                try:
                    await task
                except asyncio.CancelledError:
                    LOGGER.info("Task cancelled successfully.")
                return
            await asyncio.sleep(0.1)
            counter += 0.1
            if timeout is not None and counter >= timeout:
                LOGGER.info(f"Request processing exceeded timeout of {timeout} seconds, cancelling the task...")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    LOGGER.info("Task cancelled successfully due to timeout.")
                return JSONResponse({"error": "Request processing exceeded timeout limit."}, status_code=504)
        result = await task
    
    # delete the temporary image after processing
    try:
        image_path.unlink()
    except Exception as e:
        LOGGER.error(f"Failed to delete temporary image {image_path}: {e}")

    return JSONResponse(result) if return_detailed else JSONResponse({f"polished_{task_name}_base64_jpg": result[result["eval_image_field"]]})


async def eval_image(
    data: dict,
    task_name: str,
    model_name: str,
    return_detailed: bool,
    auth_token: str,
):

    LOGGER.info(f"Evaluating {task_name} with model_name: {model_name}")
    result = await get_score_for_image_referenced(
        sample_data=data, task_name=task_name, model_name=model_name, api_key=auth_token
    )

    if not return_detailed:
        for key in list(result.keys()):
            if not key.endswith("_outcome") or not key.endswith("_reasoning"):
                result.pop(key)

    return JSONResponse(result)


@app.post('/eval_diagram')
async def eval_diagram(
    request: Request,
    image_base64: str = Body(..., description="The input diagram to be evaluated, provided as a base64-encoded string of the image in JPG format."),
    ground_truth_image_base64: str = Body(..., description="The ground truth diagram to be used as reference for evaluation, provided as a base64-encoded string of the image in JPG format."),
    method_section: str = Body(..., description="The method section of the scientific paper to visualize, provided as plain text. Markdown format is recommended.", examples=[EXAMPLE_METHOD]),
    figure_caption: str = Body(..., description="The caption of the figure to evaluate, provided as plain text. Markdown format is recommended.", examples=[EXAMPLE_CAPTION]),
    model_name: str = Body('google/gemini-3-pro-preview', description="The name of the language model to use for processing."),
    return_detailed: bool = Body(False, description="Whether to return detailed intermediate outputs from all agents in the response."),
    timeout: typing.Optional[float] = Body(None, description="Optional timeout in seconds for the entire processing of the request. If the processing time exceeds this limit, the task will be cancelled and a timeout error will be returned.", examples=[None]),
    auth_token: str = Depends(verify_token)
):

    gt_image_path = Path(__file__).parent.parent / "data" / "PaperBananaBench" / "diagram" / f"temp_gt_{auth_token.split('-')[-1]}_{int(time.time())}_{uuid4().hex}.jpg"
    gt_image_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_image_path, "wb") as f:
        f.write(base64.b64decode(ground_truth_image_base64.split(",")[-1]))

    data = {
        "content": method_section,
        "visual_intent": figure_caption,
        "eval_image_field": "target_diagram_base64_jpg",
        "target_diagram_base64_jpg": image_base64,
        "path_to_gt_image": gt_image_path.as_posix(),
    }

    task = asyncio.create_task(eval_image(
        data=data,
        task_name="diagram",
        model_name="openrouter-" + model_name,
        return_detailed=return_detailed,
        auth_token=auth_token,
    ))

    counter = 0
    while not task.done():
        if (await request.state.is_disconnected()):
            LOGGER.info("Client disconnected, cancelling the evaluation task...")
            task.cancel()
            try:
                gt_image_path.unlink()
            except Exception as e:
                LOGGER.error(f"Failed to delete temporary ground truth image {gt_image_path} after cancellation: {e}")
            try:
                await task
            except asyncio.CancelledError:
                LOGGER.info("Task cancelled successfully.")
            return
        await asyncio.sleep(0.1)
        counter += 0.1
        if timeout is not None and counter >= timeout:
            LOGGER.info(f"Request processing exceeded timeout of {timeout} seconds, cancelling the evaluation task...")
            task.cancel()
            try:
                gt_image_path.unlink()
            except Exception as e:
                LOGGER.error(f"Failed to delete temporary ground truth image {gt_image_path} after cancellation: {e}")
            try:
                await task
            except asyncio.CancelledError:
                LOGGER.info("Task cancelled successfully due to timeout.")
            return JSONResponse({"error": "Request processing exceeded timeout limit."}, status_code=504)
    
    return await task


@app.post('/eval_plot')
async def eval_plot(
    request: Request,
    image_base64: str = Body(..., description="The input plot to be evaluated, provided as a base64-encoded string of the image in JPG format."),
    ground_truth_image_base64: str = Body(..., description="The ground truth plot to be used as reference for evaluation, provided as a base64-encoded string of the image in JPG format."),
    input_data: dict = Body(..., description="The original input data used for generating the plot, provided as a JSON object.", examples=[EXAMPLE_INPUT_DATA]),
    figure_caption: str = Body(..., description="The caption of the figure to evaluate, provided as plain text. Markdown format is recommended.", examples=[EXAMPLE_PLOT_CAPTION]),
    model_name: str = Body('google/gemini-3-pro-preview', description="The name of the language model to use for processing."),
    return_detailed: bool = Body(False, description="Whether to return detailed intermediate outputs from all agents in the response."),
    timeout: typing.Optional[float] = Body(None, description="Optional timeout in seconds for the entire processing of the request. If the processing time exceeds this limit, the task will be cancelled and a timeout error will be returned.", examples=[None]),
    auth_token: str = Depends(verify_token)
):
    
    gt_image_path = Path(__file__).parent.parent / "data" / "PaperBananaBench" / "plot" / f"temp_gt_{auth_token.split('-')[-1]}_{int(time.time())}_{uuid4().hex}.jpg"
    gt_image_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_image_path, "wb") as f:
        f.write(base64.b64decode(ground_truth_image_base64.split(",")[-1]))

    data = {
        "content": input_data,
        "visual_intent": figure_caption,
        "eval_image_field": "target_plot_base64_jpg",
        "target_plot_base64_jpg": image_base64,
        "path_to_gt_image": gt_image_path.as_posix(),
    }

    task = asyncio.create_task(eval_image(
        data=data,
        task_name="plot",
        model_name="openrouter-" + model_name,
        return_detailed=return_detailed,
        auth_token=auth_token,
    ))

    counter = 0
    while not task.done():
        if (await request.state.is_disconnected()):
            LOGGER.info("Client disconnected, cancelling the evaluation task...")
            task.cancel()
            try:
                gt_image_path.unlink()
            except Exception as e:
                LOGGER.error(f"Failed to delete temporary ground truth image {gt_image_path} after cancellation: {e}")
            try:
                await task
            except asyncio.CancelledError:
                LOGGER.info("Task cancelled successfully.")
            return
        await asyncio.sleep(0.1)
        counter += 0.1
        if timeout is not None and counter >= timeout:
            LOGGER.info(f"Request processing exceeded timeout of {timeout} seconds, cancelling the evaluation task...")
            task.cancel()
            try:
                gt_image_path.unlink()
            except Exception as e:
                LOGGER.error(f"Failed to delete temporary ground truth image {gt_image_path} after cancellation: {e}")
            try:
                await task
            except asyncio.CancelledError:
                LOGGER.info("Task cancelled successfully due to timeout.")
            return JSONResponse({"error": "Request processing exceeded timeout limit."}, status_code=504)
    
    return await task


@app.get('/', include_in_schema=False)
async def redirect():

    # return ORJSONResponse({'title': app.title, 'description': app.description, 'version': app.version})
    return RedirectResponse(app.root_path+'/docs')