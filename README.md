## Setup

### Create virtual environment

Python:
```bash
uv venv --python=3.11
source .venv/bin/activate
```
OR
Conda:
```bash
conda create -n my-env python=3.11
conda activate my-env
```

### Install requirements (run these commands in this order!!):
```bash
uv venv --python=3.11
source .venv/bin/activate
cd catanatron && uv pip install -r all-requirements.txt
cd .. && uv pip install -r requirements.txt
uv pip install -e .
cd catanatron/catanatron_core && uv pip install -e .
cd ../catanatron_experimental && uv pip install -e .
cd ../catanatron_gym && uv pip install -e .
cd ../catanatron_server && uv pip install -e .
python testing.py
```

### Test:
```bash
python testing.py
```

# Set API Keys
Add Your Desired LLM to the /agents/base_llm.py file
If you are running an Evolver, you must set your credentials in the creator_agent.py, and llm_tools.py

# strategy-game-agents

NEW WAY: Add to cli_players.py your agent (make sure to include **init**.py in directory)
`catanatron-play --players=LLM,R --num=1 --output=data/ --json`

How To View Commands
catanatron-play --help

How To View Players
catanatron-play --help-players

# View LangGraph 
langgraph dev
