<div align="center">
    <h1>Lost City</h1>
</div>

> [!NOTE]
> Learn about our history and ethos on our forum: https://lostcity.rs/t/faq-what-is-lost-city/16

> [!NOTE]
> The original Lost City source and upstream projects are published by the
> [LostCityRS GitHub organization](https://github.com/LostCityRS). This
> repository contains local engine and MCP work built on that upstream code;
> refer to the LostCityRS repositories for the original implementation and
> project updates.

This is a higher-level repository that links our other projects. You'll notice it's like home-rolled submodules (without commit references).  
Github won't include submodules in web downloads, and we have a lot of users who end up clicking download zip.

## Getting Started

> [!IMPORTANT]
> If you run into issues, please see our [common issues](#common-issues).

1. Download and extract this repo somewhere on your computer.
2. Install our [dependencies](#dependencies).
3. Open the folder you downloaded: **Run the start script and follow the on-screen prompts.** You may disregard any severity warnings you see.

Once your setup process has completed, wait for it to tell you the world has started before trying to play at: http://localhost/rs2.cgi

You can press `ctrl + c` to cancel/quit out of a terminal process.

## Dependencies

- Git CLI - Windows users: [git-scm](https://git-scm.com/)
- [NodeJS 24+](https://nodejs.org/)

> [!TIP]
> If you're using VS Code (recommended), [we have an extension to install on the marketplace.](https://marketplace.visualstudio.com/items?itemName=2004scape.runescriptlanguage)

## Workflow

**Use the start script provided** - it handles a lot of common use cases. We're trying to reduce the barrier to entry by providing an all-inclusive script.

## Agent API and MCP

The TypeScript engine exposes an authenticated Agent API at `/api/v1` on its
configured web port. The Python FastMCP adapter is in [`mcp`](mcp/README.md)
and proxies the API for agent clients. Development credentials are
`admin` / `password`.

The MCP exposes the full skill playbook, live option discovery, guarded skill
steps, and bounded automatic training loops. It observes the TypeScript health
safety state before every step and only reports level progress produced by the
game engine itself.

It also provides declarative quest planning/execution with travel, live
interactions, combat objectives, training steps, chat/dialogue actions, and
resume points for blocked objectives.

The live progression planner reads quest variables from the running engine and
uses the 2004Scape quest route as its recommendation source. It does not mark
an objective complete by writing state. Headless agents load an existing
profile save when one exists, so long-running progression can resume across
sessions.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for details.
