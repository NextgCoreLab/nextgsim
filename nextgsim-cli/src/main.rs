//! nextgsim CLI tool

mod client;
mod proc_table;
mod protocol;

use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use clap::Parser;

use client::CliClient;
use proc_table::{discover_node, list_nodes, MAX_NODE_NAME, MIN_NODE_NAME};

#[derive(Parser, Debug)]
#[command(name = "nr-cli")]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(value_name = "NODE")]
    pub node_name: Option<String>,

    #[arg(short = 'd', long = "dump")]
    pub dump: bool,

    #[arg(short = 'e', long = "exec", value_name = "COMMAND")]
    pub exec: Option<String>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ERROR: {:#}", e);
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args = Args::parse();

    if args.dump {
        return dump_nodes();
    }

    let node_name = args.node_name.as_ref()
        .context("Node name is required. Use --dump to list available nodes.")?;

    validate_node_name(node_name)?;

    let discovery = discover_node(node_name)?;
    
    if discovery.port == 0 {
        if discovery.skipped_due_to_version > 0 {
            bail!(
                "No node found with name '{}'. {} node(s) skipped due to version mismatch.",
                node_name,
                discovery.skipped_due_to_version
            );
        } else {
            bail!("No node found with name '{}'. Use --dump to list available nodes.", node_name);
        }
    }

    let client = CliClient::connect(discovery.port, node_name)
        .context("Failed to connect to node")?;

    if let Some(cmd) = &args.exec {
        execute_command(&client, cmd)
    } else {
        interactive_mode(&client)
    }
}

fn validate_node_name(name: &str) -> Result<()> {
    if name.len() < MIN_NODE_NAME {
        bail!("Node name '{}' is too short (minimum {} characters)", name, MIN_NODE_NAME);
    }
    if name.len() > MAX_NODE_NAME {
        bail!("Node name '{}' is too long (maximum {} characters)", name, MAX_NODE_NAME);
    }
    Ok(())
}

fn dump_nodes() -> Result<()> {
    let nodes = list_nodes()?;
    if nodes.is_empty() {
        println!("No running nodes found.");
    } else {
        for node in nodes {
            println!("{}", node);
        }
    }
    Ok(())
}

fn execute_command(client: &CliClient, command: &str) -> Result<()> {
    if command.len() < 3 {
        bail!("Command is too short");
    }

    match client.execute_command(command) {
        Ok((output, is_error)) => {
            if is_error {
                eprintln!("ERROR: {}", output);
                std::process::exit(1);
            } else {
                println!("{}", output);
            }
        }
        Err(e) => {
            bail!("Command execution failed: {}", e);
        }
    }
    Ok(())
}

fn interactive_mode(client: &CliClient) -> Result<()> {
    println!("Connected to {}", client.node_name());
    println!("Type 'help' for available commands, 'exit' to quit.");
    println!("{}", "-".repeat(60));

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("\x1b[1m{}> \x1b[0m", client.node_name());
        stdout.flush()?;

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                println!();
                break;
            }
            Ok(_) => {
                let cmd = line.trim();
                if cmd.is_empty() { continue; }
                if cmd == "exit" || cmd == "quit" { break; }
                if cmd == "help" {
                    print_help(client.node_name());
                    continue;
                }
                
                match client.execute_command(cmd) {
                    Ok((output, is_error)) => {
                        if is_error {
                            eprintln!("ERROR: {}", output);
                        } else if !output.is_empty() {
                            println!("{}", output);
                        }
                    }
                    Err(e) => eprintln!("ERROR: {}", e),
                }
            }
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
    }

    println!("Disconnected from {}", client.node_name());
    Ok(())
}

fn print_help(node_name: &str) {
    println!("Available commands:");
    if node_name.contains("ue") {
        println!("  status, info, ps-establish, ps-release, ps-list, deregister");
    } else if node_name.contains("gnb") {
        println!("  status, info, ue-list, amf-status");
    } else {
        println!("  status, info");
    }
    println!("  help, exit");
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_cli_parsing() { Args::command().debug_assert(); }

    #[test]
    fn test_dump_flag() {
        let args = Args::parse_from(["nr-cli", "--dump"]);
        assert!(args.dump);
    }

    #[test]
    fn test_node_name() {
        let args = Args::parse_from(["nr-cli", "gnb1"]);
        assert_eq!(args.node_name, Some("gnb1".to_string()));
    }

    #[test]
    fn test_exec_command() {
        let args = Args::parse_from(["nr-cli", "ue1", "-e", "status"]);
        assert_eq!(args.exec, Some("status".to_string()));
    }

    #[test]
    fn test_validate_node_name_too_short() {
        assert!(validate_node_name("ab").is_err());
    }

    #[test]
    fn test_validate_node_name_too_long() {
        let long_name = "a".repeat(MAX_NODE_NAME + 1);
        assert!(validate_node_name(&long_name).is_err());
    }

    #[test]
    fn test_validate_node_name_valid() {
        assert!(validate_node_name("gnb1").is_ok());
    }
}
