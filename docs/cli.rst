CLI reference
=============

The ``llmeval`` command is the primary interface for running suites, browsing
results, and starting the dashboard server.

.. code-block:: text

   Usage: llmeval [OPTIONS] COMMAND [ARGS]...

     LLM evaluation and regression testing framework.

   Commands:
     version  Print the installed llmeval version.
     run      Run a test suite, score outputs with LLM-as-judge, and print results.
     show     Display a stored run by its run ID or unique prefix.
     diff     Compare two stored runs side by side.
     list     List stored suite runs, most recent first.
     serve    Start the dashboard API server.

----

``llmeval run``
---------------

Run a test suite against a model, score outputs with an LLM-as-judge, and
print a formatted pass/fail report.

.. code-block:: bash

   llmeval run --suite SUITE_PATH [OPTIONS]

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--suite``, ``-s``
     - **(Required)** Path to a YAML or JSON test suite file.
   * - ``--model``, ``-m``
     - Override the model specified in the suite (e.g. ``gpt-4o``).
   * - ``--tag``, ``-t``
     - Only run tests with this tag. Repeatable.
   * - ``--concurrency``, ``-c``
     - Maximum simultaneous model API calls. Default: 5.
   * - ``--no-save``
     - Skip persisting the run to storage.
   * - ``--db``
     - SQLite database path. Overrides ``LLMEVAL_DB_PATH``.

**Exit codes**

- ``0`` — all tests passed
- ``1`` — one or more tests failed or errored
- ``2`` — configuration or I/O error

----

``llmeval list``
----------------

List stored suite runs, most recent first.

.. code-block:: bash

   llmeval list [OPTIONS]

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--suite``, ``-s``
     - Filter by exact suite name.
   * - ``--limit``, ``-n``
     - Maximum number of runs to show. Default: 10.
   * - ``--db``
     - SQLite database path.

----

``llmeval show``
----------------

Display a stored run by its run ID or a unique prefix.

.. code-block:: bash

   llmeval show RUN_ID [--db PATH]

The first 8 characters of a run ID are usually enough — use ``llmeval list``
to browse available IDs. An error is raised if the prefix is ambiguous.

----

``llmeval diff``
----------------

Compare two stored runs side by side and highlight regressions and
improvements per test.

.. code-block:: bash

   llmeval diff RUN_ID_A RUN_ID_B [--db PATH]

``RUN_ID_A`` is the baseline; ``RUN_ID_B`` is the candidate.

----

``llmeval serve``
-----------------

Start the FastAPI dashboard API server. Requires the ``[server]`` extras to
be installed (``pip install "llmeval[server]"``).

.. code-block:: bash

   llmeval serve [OPTIONS]

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--host``
     - Bind address. Default: ``127.0.0.1``.
   * - ``--port``, ``-p``
     - Port to listen on. Default: ``8000``.
   * - ``--db``
     - SQLite database path.
   * - ``--reload``
     - Enable auto-reload (development only).

----

``llmeval version``
-------------------

Print the installed llmeval version.

.. code-block:: bash

   llmeval version
