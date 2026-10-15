#!/usr/bin/env python3
"""Fix generator.py to use proper ACL redaction processor."""

import sys

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if the function already exists - we need to replace it
    old_func_start = '# --- Redaction for retrieved context (F-03) ---'
    if old_func_start not in content:
        print("Redaction section already fixed")
        return
    
    # Find and replace the entire _apply_redaction_to_context method
    lines = content.split('\n')
    
    new_lines = []
    skip_until_next_method = False
    
    for i, line in enumerate(lines):
        if '_apply_redaction_to_context' in line:
            # Skip this function definition (will be replaced)
            continue
        
        # Look for the process_text import inside - we need to change it
        if 'from security.acl import process_text' in line and i > 0 and '_apply_redaction' in lines[i-5:i]:
            new_lines.append('    from security.redaction import process_text as redact')
            continue
        
        # Look for the actual call with get_settings() - remove that pattern
        if 'get_settings().acl_redaction_processor(process_text, chunk.text' in line:
            new_lines.append(line.replace(
                'get_settings().acl_redaction_processor(process_text, chunk.text',
                'process_text'
            ))
            continue
        
        # Look for the function call that should use redact instead
        if 'cleaned, redactions, severity = process_text(chunk.text, mode="redact")' in line:
            new_lines.append(line)  # Keep this line but it will now work correctly with imported redact
            continue
        
        if 'logger.warning("Chunk %s empty after redaction"' in line:
            # This is part of the method we want to keep
            new_lines.append(line)
            continue
    
    # Actually, let's just rewrite the entire function properly
    replacement = '''# --- Redaction for retrieved context (F-03) ---
def _apply_redaction_to_context(self, chunks: Sequence[RetrievedChunk]) -> str:
    """
    Apply redaction to all retrieved chunks before passing to LLM.

    Fails closed — empty content after redaction returns empty context.
    """
    from security.redaction import process_text as redact
    
    context_parts = []
    for chunk in chunks:
        if not chunk.text or isinstance(chunk.text, bytes):
            continue
        try:
            cleaned, redactions, severity = redact(chunk.text, mode="redact")
            if not cleaned or len(cleaned.strip()) == 0:
                logger.warning("Chunk %s empty after redaction", chunk.doc_id)
                continue
            context_parts.append({
                "text": cleaned,
                "doc_ids": [chunk.doc_id],
                "redactions_applied": redactions,
                "max_severity": severity,
                "page_content": chunk.page_content if hasattr(chunk, 'page_content') else None,
            })
        except Exception as e:
            logger.error("Redaction error for chunk %s: %s", chunk.doc_id, str(e))
            continue

    return "\\n\\n".join([f"{item['doc_ids']}: {item['text']}"
                       + (f" [redactions: {len(item['redactions_applied'])}]")
                       for item in context_parts])'''
    
    # Find the start and end of the old function to replace it
    found_start = False
    
    new_content_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if '# --- Redaction for retrieved context (F-03) ---' in line:
            # Skip until we find the next function or class definition
            new_content_lines.append(replacement)
            i += 1
            while i < len(lines):
                if lines[i].strip() and not lines[i].startswith(' ') and not lines[i].startswith('\t'):
                    break
                if 'def _apply_redaction_to_context' in lines[i]:
                    # Found the function definition, now skip all its lines until next def/class or __all__
                    i += 1
                    while i < len(lines):
                        current = lines[i]
                        if not current.strip():
                            i += 1
                            continue
                        if current.strip().startswith('def ') and 'redaction' not in current:
                            break
                        if current.strip().startswith('# ---') and 'Redaction for retrieved context (F-03)' in current:
                            # Found the start marker again, skip one more line
                            i += 1
                            continue
                        i += 1
                    break
        else:
            new_content_lines.append(line)
        
        i += 1
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(new_content_lines))
    
    print(f"Fixed {filepath}")

if __name__ == '__main__':
    fix_file(sys.argv[1])