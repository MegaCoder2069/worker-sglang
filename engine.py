import subprocess
import time
import requests
import openai
import asyncio
import aiohttp
import os


DEEPSEEK_V4_FLASH_FP8_MODEL = "sgl-project/DeepSeek-V4-Flash-FP8"
DEEPSEEK_V4_SERVED_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
DEEPSEEK_V4_CONTEXT_LENGTH = "400000"
DEFAULT_SGLANG_PRESET = "deepseek-v4-flash-fp8"
DEEPEP_96_SMS_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'
DISABLED_ENV_VALUES = {"0", "false", "no", "off", "none", "disabled"}


def _is_disabled(value):
    return value.strip().lower() in DISABLED_ENV_VALUES


def _env_default(name, value):
    if os.getenv(name) in (None, ""):
        os.environ[name] = value


def _configured_value(name):
    value = os.getenv(name)
    if value is None or value == "" or _is_disabled(value):
        return None
    return value


def _configured_int(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        return default


class SGlangEngine:
    def __init__(
        self,
        model=None,
        host=None,
        port=None,
    ):
        self.preset = os.getenv("SGLANG_PRESET", DEFAULT_SGLANG_PRESET).strip().lower()
        self.model = (
            model
            or _configured_value("SERVED_MODEL_NAME")
            or _configured_value("MODEL_NAME")
            or self._preset_served_model()
        )
        self.host = host or os.getenv("HOST", "0.0.0.0")
        self.port = int(port or os.getenv("PORT", 30000))
        self.bind_url = f"http://{self.host}:{self.port}"
        self.client_host = os.getenv(
            "SGLANG_CLIENT_HOST",
            "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host,
        )
        self.base_url = f"http://{self.client_host}:{self.port}"
        self.process = None

    def _preset_served_model(self):
        if self.preset == DEFAULT_SGLANG_PRESET:
            return DEEPSEEK_V4_SERVED_MODEL
        return None

    def _preset_enabled(self):
        return not _is_disabled(self.preset)

    def _detect_deepseek_v4_hardware(self):
        configured = os.getenv("DEEPSEEK_V4_HARDWARE", "h200").strip().lower()
        if configured in ("h100", "h200"):
            return configured

        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).lower()
        except (FileNotFoundError, subprocess.SubprocessError):
            return "h200"

        if "h100" in output:
            return "h100"
        if "h200" in output:
            return "h200"
        return "h200"

    def _deepseek_v4_preset(self):
        if self.preset != DEFAULT_SGLANG_PRESET:
            return {}, [], {}

        hardware = self._detect_deepseek_v4_hardware()
        tp = "8" if hardware == "h100" else "4"
        recipe = os.getenv("DEEPSEEK_V4_RECIPE", "balanced").strip().lower()

        options = {
            "MODEL_NAME": DEEPSEEK_V4_FLASH_FP8_MODEL,
            "SERVED_MODEL_NAME": DEEPSEEK_V4_SERVED_MODEL,
            "CONTEXT_LENGTH": DEEPSEEK_V4_CONTEXT_LENGTH,
            "TP": tp,
            "TOOL_CALL_PARSER": "deepseekv4",
            "REASONING_PARSER": "deepseek-v4",
        }
        boolean_flags = ["TRUST_REMOTE_CODE"]
        environment = {"SGLANG_DSV4_FP4_EXPERTS": "0"}

        if recipe == "low-latency":
            options.update(
                {
                    "SPECULATIVE_ALGO": "EAGLE",
                    "SPECULATIVE_NUM_STEPS": "3",
                    "SPECULATIVE_EAGLE_TOPK": "1",
                    "SPECULATIVE_NUM_DRAFT_TOKENS": "4",
                }
            )
        elif recipe == "max-throughput":
            options.update(
                {
                    "DP": tp,
                    "MOE_A2A_BACKEND": "deepep",
                    "DEEPEP_CONFIG": DEEPEP_96_SMS_CONFIG,
                }
            )
            boolean_flags.append("ENABLE_DP_ATTENTION")
            environment["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "256"
            if hardware == "h200":
                options.update(
                    {
                        "CUDA_GRAPH_MAX_BS": "128",
                        "MAX_RUNNING_REQUESTS": "256",
                    }
                )
        else:
            options.update(
                {
                    "DP": tp,
                    "MOE_A2A_BACKEND": "deepep",
                    "SPECULATIVE_ALGO": "EAGLE",
                    "SPECULATIVE_NUM_STEPS": "1",
                    "SPECULATIVE_EAGLE_TOPK": "1",
                    "SPECULATIVE_NUM_DRAFT_TOKENS": "2",
                    "DEEPEP_CONFIG": DEEPEP_96_SMS_CONFIG,
                }
            )
            boolean_flags.append("ENABLE_DP_ATTENTION")
            environment["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "256"
            if hardware == "h200":
                options.update(
                    {
                        "CUDA_GRAPH_MAX_BS": "128",
                        "MAX_RUNNING_REQUESTS": "128",
                    }
                )

        return options, boolean_flags, environment

    def _option_value(self, env_vars, preset_options):
        for env_var in env_vars:
            value = os.getenv(env_var)
            if value is None or value == "":
                continue
            if _is_disabled(value):
                return None
            return value
        for env_var in env_vars:
            value = preset_options.get(env_var)
            if value is not None:
                return value
        return None

    def _flag_enabled(self, flag, preset_flags):
        value = os.getenv(flag)
        if value is not None and value != "":
            return value.lower() in ("true", "1", "yes", "on")
        return flag in preset_flags

    def build_command(self):
        preset_options, preset_flags, preset_environment = (
            self._deepseek_v4_preset() if self._preset_enabled() else ({}, [], {})
        )

        for name, value in preset_environment.items():
            _env_default(name, value)

        command = [
            os.getenv("SGLANG_COMMAND", "sglang"),
            os.getenv("SGLANG_SUBCOMMAND", "serve"),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        # Dictionary of all possible options and their corresponding env var names
        options = [
            (("MODEL_NAME",), "--model-path"),
            (("TOKENIZER_PATH",), "--tokenizer-path"),
            (("TOKENIZER_MODE",), "--tokenizer-mode"),
            (("LOAD_FORMAT",), "--load-format"),
            (("DTYPE",), "--dtype"),
            (("CONTEXT_LENGTH",), "--context-length"),
            (("QUANTIZATION",), "--quantization"),
            (("SERVED_MODEL_NAME",), "--served-model-name"),
            (("CHAT_TEMPLATE",), "--chat-template"),
            (("MEM_FRACTION_STATIC",), "--mem-fraction-static"),
            (("MAX_RUNNING_REQUESTS",), "--max-running-requests"),
            (("MAX_TOTAL_TOKENS",), "--max-total-tokens"),
            (("CHUNKED_PREFILL_SIZE",), "--chunked-prefill-size"),
            (("MAX_PREFILL_TOKENS",), "--max-prefill-tokens"),
            (("SCHEDULE_POLICY",), "--schedule-policy"),
            (("SCHEDULE_CONSERVATIVENESS",), "--schedule-conservativeness"),
            (("TP", "TENSOR_PARALLEL_SIZE"), "--tp"),
            (("DP", "DATA_PARALLEL_SIZE"), "--dp"),
            (("STREAM_INTERVAL",), "--stream-interval"),
            (("RANDOM_SEED",), "--random-seed"),
            (("LOG_LEVEL",), "--log-level"),
            (("LOG_LEVEL_HTTP",), "--log-level-http"),
            (("API_KEY",), "--api-key"),
            (("FILE_STORAGE_PATH",), "--file-storage-path"),
            (("LOAD_BALANCE_METHOD",), "--load-balance-method"),
            (("ATTENTION_BACKEND",), "--attention-backend"),
            (("SAMPLING_BACKEND",), "--sampling-backend"),
            (("MOE_A2A_BACKEND",), "--moe-a2a-backend"),
            (("MOE_RUNNER_BACKEND",), "--moe-runner-backend"),
            (("CUDA_GRAPH_MAX_BS",), "--cuda-graph-max-bs"),
            (("DEEPEP_CONFIG",), "--deepep-config"),
            (("SPECULATIVE_ALGO", "SPECULATIVE_ALGORITHM"), "--speculative-algo"),
            (("SPECULATIVE_NUM_STEPS",), "--speculative-num-steps"),
            (("SPECULATIVE_EAGLE_TOPK",), "--speculative-eagle-topk"),
            (("SPECULATIVE_NUM_DRAFT_TOKENS",), "--speculative-num-draft-tokens"),
            (("PAGE_SIZE",), "--page-size"),
            (("KV_CACHE_DTYPE",), "--kv-cache-dtype"),
            (("TOOL_CALL_PARSER",), "--tool-call-parser"),
            (("REASONING_PARSER",), "--reasoning-parser"),
        ]

        # Boolean flags
        boolean_flags = [
            "SKIP_TOKENIZER_INIT",
            "TRUST_REMOTE_CODE",
            "LOG_REQUESTS",
            "SHOW_TIME_COST",
            "DISABLE_RADIX_CACHE",
            "DISABLE_CUDA_GRAPH",
            "DISABLE_OUTLINES_DISK_CACHE",
            "ENABLE_TORCH_COMPILE",
            "ENABLE_P2P_CHECK",
            "ENABLE_FLASHINFER_MLA",
            "TRITON_ATTENTION_REDUCE_IN_FP32",
            "ENABLE_DP_ATTENTION",
            "ALLOW_AUTO_TRUNCATE",
            "ENABLE_METRICS",
        ]

        # Add options from environment variables only if they are set
        for env_vars, option in options:
            value = self._option_value(env_vars, preset_options)
            if value:
                command.extend([option, value])

        # Add boolean flags only if they are set to true
        for flag in boolean_flags:
            if self._flag_enabled(flag, preset_flags):
                command.append(f"--{flag.lower().replace('_', '-')}")

        return command

    def start_server(self):
        command = self.build_command()
        print(f"Starting SGLang with command: {' '.join(command)}", flush=True)
        self.process = subprocess.Popen(command, stdout=None, stderr=None)
        print(f"Server started with PID: {self.process.pid}", flush=True)

    def wait_for_server(self, timeout=None, interval=None):
        timeout = timeout if timeout is not None else _configured_int("SERVER_START_TIMEOUT", 3600)
        interval = interval if interval is not None else _configured_int("SERVER_READY_INTERVAL", 5)
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(
                    f"SGLang server exited before readiness check passed "
                    f"(exit code {self.process.returncode})."
                )
            try:
                response = requests.get(f"{self.base_url}/v1/models")
                if response.status_code == 200:
                    print("Server is ready!", flush=True)
                    return True
            except requests.RequestException:
                pass
            time.sleep(interval)
        raise TimeoutError("Server failed to start within the timeout period.")

    def shutdown(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Server shut down.")


class OpenAIRequest:
    def __init__(self, base_url="http://0.0.0.0:30000/v1", api_key="EMPTY"):
        self.client = openai.Client(base_url=base_url, api_key=api_key)

    async def request_chat_completions(
        self,
        model="default",
        messages=None,
        max_tokens=100,
        stream=False,
        frequency_penalty=0.0,
        n=1,
        stop=None,
        temperature=1.0,
        top_p=1.0,
    ):
        if messages is None:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "List 3 countries and their capitals."},
            ]

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=stream,
            frequency_penalty=frequency_penalty,
            n=n,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        )

        if stream:
            async for chunk in response:
                yield chunk.to_dict()
        else:
            yield response.to_dict()

    async def request_completions(
        self,
        model="default",
        prompt="The capital of France is",
        max_tokens=100,
        stream=False,
        frequency_penalty=0.0,
        n=1,
        stop=None,
        temperature=1.0,
        top_p=1.0,
    ):
        response = self.client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            stream=stream,
            frequency_penalty=frequency_penalty,
            n=n,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
        )

        if stream:
            async for chunk in response:
                yield chunk.to_dict()
        else:
            yield response.to_dict()

    async def get_models(self):
        response = await self.client.models.list()
        return response
