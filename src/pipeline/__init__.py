import sys
import importlib
import importlib.abc
import importlib.util

class _LegacyAliasLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("pipeline.Pipeline.") or fullname == "pipeline.Pipeline":
            real_name = "pipeline" + fullname[len("pipeline.Pipeline"):]
            real_mod = importlib.import_module(real_name)
            sys.modules[fullname] = real_mod
            spec = importlib.util.spec_from_loader(fullname, self)
            if hasattr(real_mod, "__path__"):
                spec.submodule_search_locations = real_mod.__path__
            return spec
        elif fullname.startswith("Temporal.Pipeline.") or fullname in ("Temporal", "Temporal.Pipeline"):
            real_name = "pipeline" + fullname[len("Temporal.Pipeline"):] if fullname.startswith("Temporal.Pipeline") else "pipeline"
            real_mod = importlib.import_module(real_name)
            sys.modules[fullname] = real_mod
            spec = importlib.util.spec_from_loader(fullname, self)
            if hasattr(real_mod, "__path__"):
                spec.submodule_search_locations = real_mod.__path__
            return spec
        return None

    def create_module(self, spec):
        return sys.modules.get(spec.name)

    def exec_module(self, module):
        pass

if not any(isinstance(finder, _LegacyAliasLoader) for finder in sys.meta_path):
    sys.meta_path.insert(0, _LegacyAliasLoader())




