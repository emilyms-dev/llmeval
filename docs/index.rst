llmeval
=======

An open-source Python framework for behavioral testing and regression
detection of LLM-powered applications.

Define prompt test suites in YAML, run them against any OpenAI or Anthropic
model, score outputs with an LLM-as-judge, and catch regressions across model
versions — all integrated with GitHub Actions and a React dashboard.

.. code-block:: bash

   pip install llmeval
   llmeval run --suite my_suite.yaml

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   suite_format
   cli

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog
