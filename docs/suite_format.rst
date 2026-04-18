Test suite format
=================

Test suites are defined in YAML (or JSON) files. The schema is validated by
Pydantic on load — any missing or invalid fields produce a clear error message.

Full example
------------

.. code-block:: yaml

   suite:
     name: "Customer Support Bot — Tone Tests"
     version: "1.0.0"
     model: "claude-sonnet-4-20250514"
     judge_model: "claude-sonnet-4-20250514"

   tests:
     - id: "tone-empathy-001"
       description: "Response should be empathetic when user is frustrated"
       prompt: "I've been waiting 3 weeks for my order and nobody will help me!"
       rubric:
         criteria:
           - name: "empathy"
             description: "Response acknowledges the user's frustration"
             weight: 0.4
           - name: "actionability"
             description: "Response offers a concrete next step"
             weight: 0.4
           - name: "tone"
             description: "Response is professional and not defensive"
             weight: 0.2
         passing_threshold: 0.75
       tags: ["tone", "empathy", "regression-critical"]

``suite`` block
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Field
     - Required
     - Description
   * - ``name``
     - Yes
     - Human-readable suite name. Displayed in CLI output and the dashboard.
   * - ``version``
     - Yes
     - Semantic version string. Included in stored run records.
   * - ``model``
     - Yes
     - Default model for test execution (e.g. ``claude-sonnet-4-20250514``,
       ``gpt-4o``). Can be overridden per-run with ``--model``.
   * - ``judge_model``
     - Yes
     - Model used for LLM-as-judge scoring. Can differ from the test model.

``tests`` block
---------------

Each entry in the ``tests`` list is a single test case.

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Field
     - Required
     - Description
   * - ``id``
     - Yes
     - Unique identifier. Used as the key when comparing runs with ``diff``.
   * - ``prompt``
     - Yes
     - The prompt sent verbatim to the model.
   * - ``description``
     - No
     - Human-readable description of what the test checks.
   * - ``rubric``
     - Yes
     - Scoring rubric (see below).
   * - ``tags``
     - No
     - List of string tags. Use ``--tag`` to run only a subset.

Rubric
------

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Field
     - Required
     - Description
   * - ``criteria``
     - Yes
     - List of scoring criteria (see below). Weights must sum to 1.0.
   * - ``passing_threshold``
     - Yes
     - Minimum weighted score (0.0–1.0) for the test to be marked as passed.

Each criterion:

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Field
     - Required
     - Description
   * - ``name``
     - Yes
     - Short identifier for the criterion (e.g. ``"empathy"``).
   * - ``description``
     - Yes
     - Sent to the judge model as the scoring instruction.
   * - ``weight``
     - Yes
     - Relative weight of this criterion. All weights in a rubric must sum to 1.0.

Supported models
----------------

Model prefixes are used to select the correct API adapter:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Prefix
     - Provider
   * - ``claude-``
     - Anthropic (requires ``ANTHROPIC_API_KEY``)
   * - ``gpt-``, ``o1-``, ``o3-``
     - OpenAI (requires ``OPENAI_API_KEY``)

Tag filtering
-------------

Run only tests that have a specific tag:

.. code-block:: bash

   llmeval run --suite suite.yaml --tag regression-critical
   llmeval run --suite suite.yaml --tag tone --tag empathy
