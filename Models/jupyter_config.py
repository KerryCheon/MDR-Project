c = get_config()

# MCP server config has moved to jupyter_config/jupyter_server_config.d/jupyter_mcp.json
# (loaded via JUPYTER_CONFIG_PATH environment variable)
# 
# Launch with:
#   $env:JUPYTER_CONFIG_PATH="jupyter_config"; uv run --with jupyter jupyter lab
#
# Password/token config stays in ~/.jupyter/jupyter_server_config.json (not in repo)



#   "jupyterlab_commands_toolkit.tools:list_all_commands",
#   "jupyterlab_commands_toolkit.tools:execute_command",