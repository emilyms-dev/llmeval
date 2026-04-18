Model adapters
==============

Adapters abstract OpenAI and Anthropic APIs behind a common interface.
Use :func:`~llmeval.models.create_adapter` to instantiate the correct adapter
from a model name string.

.. autofunction:: llmeval.models.create_adapter

Base adapter
------------

.. automodule:: llmeval.models.base
   :members:
   :undoc-members:
   :show-inheritance:

Anthropic adapter
-----------------

.. automodule:: llmeval.models.anthropic_adapter
   :members:
   :undoc-members:
   :show-inheritance:

OpenAI adapter
--------------

.. automodule:: llmeval.models.openai_adapter
   :members:
   :undoc-members:
   :show-inheritance:
