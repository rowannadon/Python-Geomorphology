"""Global context for node graph execution."""

from typing import Dict, Any, Optional


class NodeGraphContext:
    """Global context shared across all nodes in the graph."""
    
    def __init__(self):
        self._properties = {
            'dim': 1024,  # Default resolution
        }
        self._map_properties_node = None
    
    def set_map_properties_node(self, node):
        """Set the reference to the map properties node."""
        self._map_properties_node = node
    
    def get_resolution(self) -> int:
        """Get the current resolution from the map properties node."""
        if self._map_properties_node is not None:
            dim_str = self._map_properties_node.get_property('dimension')
            return int(dim_str)
        return self._properties['dim']
    
    def get_property(self, name: str, default: Any = None) -> Any:
        """Get a global property value."""
        if name == 'dim':
            return self.get_resolution()
        return self._properties.get(name, default)
    
    def set_property(self, name: str, value: Any):
        """Set a global property value."""
        self._properties[name] = value


# Global context instance
_global_context = NodeGraphContext()


def get_global_context() -> NodeGraphContext:
    """Get the global node graph context."""
    return _global_context