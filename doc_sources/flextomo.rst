Modules
================

flexTOMO includes three modules: projector, phantom and model.

The projector module provides access to forward- / back-projection operators and whole reconstruction algorithms. It supports out-of-memory input arrays (numpy.memmap) and can be accelerated through the use of subsets.

Modules model and phantom can be used to simulate both monochromatic and polychromatic conde-beam tomographic acquisition.

flextomo.projector
-------------------------

.. automodule:: flextomo.projector
    :members:
    :undoc-members:
    :show-inheritance:

flextomo.phantom
-----------------------

.. automodule:: flextomo.phantom
    :members:
    :undoc-members:
    :show-inheritance:

flextomo.model
---------------------

.. automodule:: flextomo.model
    :members:
    :undoc-members:
    :show-inheritance:

Module contents
---------------

.. automodule:: flextomo
    :members:
    :undoc-members:
    :show-inheritance:

Backend
-------

The backend of the projector module (and by extension all of flexTOMO and flexCALC) is provided by the GPU-accelerated ASTRA Toolbox. It is encapsulated in four functions: :py:meth:`flextomo.projector.forwardproject`, :py:meth:`flextomo.projector.backproject`, :py:meth:`flexdata.geometry.astra_volume_geom`, and :py:meth:`flexdata.geometry.astra_projection_geom`.
