from langgraph.graph import END, StateGraph


def build_agent(config, tools):
    def planner_node(state):
        return {
            "bucket": state.get("bucket", config.minio_bucket),
            "prefix": state.get("prefix", config.minio_prefix),
            "local_dir": state.get("local_dir", config.local_dir),
            "execution_mode": "single_thread",
        }

    def tool_executor_node(state):
        try:
            download_result = tools["downloader_from_datalake"](
                {"bucket": state["bucket"], "prefix": state["prefix"], "local_dir": state["local_dir"]}
            )
            find_result = tools["find_excel_files"]({"local_dir": state["local_dir"]})
            excel_files = find_result.get("excel_files", [])
            read_result = tools["read_excel_files_single_thread"]({"excel_files": excel_files})
            full_text_result = tools["read_excel_full_text_dict"]({"excel_files": excel_files})
            keys_result = tools["identify_keys_in_sections_llm"](
                {"excel_full_text_dict": full_text_result.get("excel_full_text_dict", {})}
            )
            schema_result = tools["build_common_schema"](
                {"identified_section_keys": keys_result.get("identified_section_keys", {})}
            )
            fill_result = tools["fill_schema_values_llm"](
                {
                    "excel_full_text_dict": full_text_result.get("excel_full_text_dict", {}),
                    "common_schema": schema_result.get("common_schema", {}),
                }
            )

            final_message = (
                f"{download_result.get('message', '')} "
                f"{find_result.get('message', '')} "
                f"{read_result.get('message', '')} "
                f"{full_text_result.get('message', '')} "
                f"{keys_result.get('message', '')} "
                f"{schema_result.get('message', '')} "
                f"{fill_result.get('message', '')}"
            ).strip()

            return {
                "excel_files": excel_files,
                "excel_summaries": read_result.get("excel_summaries", []),
                "excel_full_text_dict": full_text_result.get("excel_full_text_dict", {}),
                "identified_section_keys": keys_result.get("identified_section_keys", {}),
                "common_schema": schema_result.get("common_schema", {}),
                "products_dict": fill_result.get("products_dict", {}),
                "product_table_rows": fill_result.get("product_table_rows", []),
                "message": final_message,
                "openai_configured": bool(config.openai_api_key),
                "execution_mode": "single_thread",
            }
        except Exception as exc:
            return {"error": f"Tool execution failed: {exc}"}

    graph_builder = StateGraph(dict)
    graph_builder.add_node("planner", planner_node)
    graph_builder.add_node("tool_executor", tool_executor_node)
    graph_builder.set_entry_point("planner")
    graph_builder.add_edge("planner", "tool_executor")
    graph_builder.add_edge("tool_executor", END)
    return graph_builder.compile()
