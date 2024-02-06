#!/bin/bash

# Define the commands and their subcommands
dbs_list=(
    "pinecone"
    "qdrant"
    "milvus"
    "vertexai_vectorsearch"
    "kdbai"
)
# define export_vdf_list programmatically as "export_vdf <db_name>"
commands=()
for db in "${dbs_list[@]}"; do
    commands+=("export_vdf $db")
done
# same for import for each db
for db in "${dbs_list[@]}"; do
    commands+=("import_vdf $db")
done

# Create the docs/ folder if it doesn't exist
mkdir -p docs

# Run the help command for each command and subcommand
for command in "${commands[@]}"; do
    # Extract the command name
    
    # Create a separate text file for each command
    output_file="docs/${command// /_}_help.txt"
    # Run the help command and save the output to the file
    echo "Help for '$command':" > "$output_file"
    $command --help >> "$output_file"
    echo "" >> "$output_file"
done

echo "Help outputs have been written to the 'docs/' folder."
