import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import pi_core.features as features
print("file:", features.__file__)
print("has attach_reference?:", hasattr(features, "attach_reference"))
if hasattr(features, "attach_reference"):
    import inspect
    print("signature:", inspect.signature(features.attach_reference))
