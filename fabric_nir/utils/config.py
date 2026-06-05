"""
配置管理模块
"""

import os
import yaml
from typing import Dict, Any, Optional


class ConfigManager:
    """
    配置管理器
    支持配置文件加载、继承和合并
    """
    
    def __init__(self, config_path: str):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        支持配置继承
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            config: 配置字典
        """
        # 加载配置文件
        with open(config_path, 'r',encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 处理继承
        if "inherit" in config:
            # 获取父配置文件路径
            parent_path = config.pop("inherit")
            
            # 如果是相对路径，转换为绝对路径
            if not os.path.isabs(parent_path):
                parent_path = os.path.join(os.path.dirname(config_path), parent_path)
            
            # 加载父配置
            parent_config = self._load_config(parent_path)
            
            # 合并配置
            config = self._merge_configs(parent_config, config)
        
        return config
    
    def _merge_configs(self, parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并配置
        子配置会覆盖父配置中的同名项
        
        Args:
            parent: 父配置
            child: 子配置
            
        Returns:
            merged: 合并后的配置
        """
        merged = parent.copy()
        
        for key, value in child.items():
            # 如果是字典，递归合并
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def save(self, path: Optional[str] = None) -> None:
        """
        保存配置
        
        Args:
            path: 保存路径，默认为原配置文件路径
        """
        if path is None:
            path = self.config_path
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # 保存配置
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        支持点分隔的键
        
        Args:
            key: 配置键，支持点分隔，如 "model.backbone.type"
            default: 默认值
            
        Returns:
            value: 配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置项
        支持点分隔的键
        
        Args:
            key: 配置键，支持点分隔，如 "model.backbone.type"
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
