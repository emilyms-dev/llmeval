Getting started
===============

Installation
------------

llmeval requires Python 3.11 or later.

.. code-block:: bash

   pip install llmeval

To use the dashboard API server, install the optional server extras:

.. code-block:: bash

   pip install "llmeval[server]"

Configuration
-------------

API keys are read from environment variables. Create a ``.env`` file in your
project root (llmeval loads it automatically on startup):

.. code-block:: bash

   ANTHROPIC_API_KEY=sk-ant-...
   OPENAI_API_KEY=sk-...

   # Optional — defaults to llmeval.db in the current directory
   LLMEVAL_DB_PATH=./llmeval.db

Quick start
-----------

1. Write a test suite YAML file (see :doc:`suite_format` for the full schema):

   .. code-block:: yaml

      suite:
        name: "My App — Smoke Tests"
        version: "1.0.0"
        model: "claude-sonnet-4-20250514"
        judge_model: "claude-sonnet-4-20250514"

      tests:
        - id: "friendly-greeting"
          prompt: "Say hello to a new user joining our platform."
          rubric:
            criteria:
              - name: "tone"
                description: "Response is warm and welcoming"
                weight: 0.6
              - name: "brevity"
                description: "Response is concise (under 3 sentences)"
                weight: 0.4
            passing_threshold: 0.75

2. Run the suite:

   .. code-block:: bash

      llmeval run --suite my_suite.yaml

3. View stored results:

   .. code-block:: bash

      llmeval list
      llmeval show <run-id>

4. Compare two runs (regression detection):

   .. code-block:: bash

      llmeval diff <baseline-run-id> <candidate-run-id>

Python API
----------

You can also drive evaluations programmatically:

.. code-block:: python

   import asyncio
   from llmeval import Judge, Runner
   from llmeval.models import create_adapter
   from llmeval.schema.test_suite import load_suite
   from llmeval.storage import SQLiteStorage

   async def main():
       suite = load_suite("my_suite.yaml")
       runner = Runner(create_adapter(suite.suite.model))
       suite_run = await runner.run(suite)

       judge = Judge(create_adapter(suite.suite.judge_model))
       suite_run = await judge.score_suite_run(suite_run, suite)

       async with SQLiteStorage("llmeval.db") as storage:
           await storage.save_run(suite_run)

       print(f"Passed: {suite_run.passed_tests}/{suite_run.total_tests}")

   asyncio.run(main())

Dashboard
---------

Start the local dashboard server:

.. code-block:: bash

   llmeval serve

Then open the React frontend (from the ``dashboard/`` directory):

.. code-block:: bash

   cd dashboard
   npm install
   npm run dev

Navigate to http://localhost:5173 to browse runs, inspect results, and
trigger new runs directly from the UI.
