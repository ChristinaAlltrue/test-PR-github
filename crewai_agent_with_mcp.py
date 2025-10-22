#  Copyright 2023-2024 AllTrue.ai Inc
#  All Rights Reserved.
#
#  NOTICE: All information contained herein is, and remains
#  the property of AllTrue.ai Incorporated. The intellectual and technical
#  concepts contained herein are proprietary to AllTrue.ai Incorporated
#  and may be covered by U.S. and Foreign Patents,
#  patents in process, and are protected by trade secret or copyright law.
#  Dissemination of this information or reproduction of this material
#  is strictly forbidden unless prior written permission is obtained
#  from AllTrue.ai Incorporated.


import asyncio

from crewai import Agent, Task, Crew, CrewOutput
from crewai_tools import MCPServerAdapter


async def run_query(query: str, server_url: str) -> CrewOutput | None:
    """
    Kick off a CrewAI agent that uses tools from a remote MCP server.

    :param query: The user's request or question that the agent should satisfy.
    :param server_url: The base URL of the remote MCP server.

    :return: The final output from the CrewAI workflow, or None if an error occurs.
    """
    server_params = {
        "url": "https://mcp.deepwiki.com/sse",  # override the user input for demonstration purposes
        "transport": "sse",  # or "streamable-http"
    }  # If the server requires authentication, we can supply additional parameters such as headers or OAuth tokens here

    try:
        with MCPServerAdapter(server_params) as tools:
            tool_names = [tool.name for tool in tools]
            print(f"Available tools from remote server: {tool_names}")

            remote_agent = Agent(
                role="Remote Tool User",
                goal="Use the remote MCP tools to answer user queries and provide helpful responses.",
                backstory=(
                    "You are connected to a remote MCP server that provides a set of tools. "
                    "Examine the user's question and decide which tool (if any) can help. "
                    "Call the tool with appropriate arguments and return a concise answer."
                ),
                tools=tools,
                reasoning=True,
                verbose=True,
                llm="gpt-4o"
            )

            doc_task = Task(
                description="Answer the following question for the user: '{query}'. "
                "You have access to remote tools; call them when helpful and include the result in your answer.",
                agent=remote_agent,
                expected_output="A concise answer to the user's question, possibly using remote tools.",
            )

            crew = Crew(
                agents=[remote_agent],
                tasks=[doc_task],
                verbose=True,
            )

            result = crew.kickoff(inputs={"query": query})
            return result
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(override=True)

    print("CrewAI Remote MCP Example")
    print("Enter a question or request and the agent will try to answer it using a remote MCP server.")
    user_query = input("Your query: ")

    # We'll use a publicly accessible MCP server like 'https://mcp.deepwiki.com/sse' or 'https://mcp.remote-mcp.com'.
    default_server = "https://mcp.deepwiki.com/sse"
    server_input = input(f"Enter MCP server URL [type 'default' to use {default_server}]: ").strip()
    server_url = default_server if server_input.lower() == "default" else server_input

    result = asyncio.run(run_query(user_query, server_url))
    if result:
        print("\nFinal output:\n")
        print(result)
    else:
        print("No result returned.  Check the server URL and try again.")


if __name__ == "__main__":
    main()
