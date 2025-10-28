"""Integration test demonstrating hash chain integrity protection."""
import sys
from pathlib import Path
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.evidence_registry import EvidenceRegistry


def demonstrate_hash_chain_protection():
    """
    Demonstrate that the hash chain protects against tampering.
    This is the scenario described in the problem statement.
    """
    print("=" * 70)
    print("HASH CHAIN INTEGRITY PROTECTION DEMONSTRATION")
    print("=" * 70)
    print()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "ledger.jsonl"
        
        # Step 1: Create a legitimate evidence chain
        print("Step 1: Creating legitimate evidence chain...")
        registry = EvidenceRegistry(storage_path=storage_path)
        
        e1_id = registry.record_evidence(
            evidence_type="policy_analysis",
            payload={"finding": "Budget allocation is appropriate"},
        )
        print(f"  ✓ Recorded entry 1: {e1_id[:16]}...")
        
        e2_id = registry.record_evidence(
            evidence_type="policy_analysis",
            payload={"finding": "Timeline is feasible"},
        )
        print(f"  ✓ Recorded entry 2: {e2_id[:16]}...")
        
        e3_id = registry.record_evidence(
            evidence_type="policy_analysis",
            payload={"finding": "Stakeholder engagement is adequate"},
        )
        print(f"  ✓ Recorded entry 3: {e3_id[:16]}...")
        print()
        
        # Step 2: Verify the legitimate chain
        print("Step 2: Verifying legitimate chain integrity...")
        is_valid, errors = registry.verify_chain_integrity()
        print(f"  Chain valid: {is_valid}")
        print(f"  Errors: {len(errors)}")
        print()
        
        # Step 3: Simulate tampering - modify previous_hash
        print("Step 3: Simulating attack - tampering with previous_hash...")
        print("  (This is the vulnerability the fix addresses)")
        
        # Read the ledger file
        with open(storage_path, 'r') as f:
            lines = f.readlines()
        
        # Display original entry 2
        entry2_original = json.loads(lines[1])
        print(f"  Original entry 2 previous_hash: {entry2_original['previous_hash'][:16]}...")
        
        # Tamper with entry 2's previous_hash using a fake hash
        FAKE_HASH = "0" * 64  # Invalid hash value to simulate tampering
        entry2_original['previous_hash'] = FAKE_HASH
        lines[1] = json.dumps(entry2_original) + '\n'
        
        print(f"  Tampered entry 2 previous_hash: {FAKE_HASH[:16]}...")
        
        # Write tampered data back
        with open(storage_path, 'w') as f:
            f.writelines(lines)
        print()
        
        # Step 4: Attempt to verify tampered chain
        print("Step 4: Verifying tampered chain...")
        print("  (The fix should detect the tampering)")
        
        # Reload registry with tampered data
        registry_tampered = EvidenceRegistry(storage_path=storage_path)
        is_valid, errors = registry_tampered.verify_chain_integrity()
        
        print(f"  Chain valid: {is_valid}")
        print(f"  Errors detected: {len(errors)}")
        
        if not is_valid and len(errors) > 0:
            print("  ✓ SUCCESS: Tampering was detected!")
            print(f"  Error message: {errors[0][:100]}...")
        else:
            print("  ✗ FAILURE: Tampering was NOT detected!")
            return False
        
        print()
        
        # Step 5: Demonstrate the fix
        print("Step 5: What the fix does...")
        print("  Before the fix:")
        print("    - verify() only checked if entry_hash matched recomputed hash")
        print("    - Did NOT verify that previous_hash matched actual prior entry")
        print("    - Attacker could change previous_hash without detection")
        print()
        print("  After the fix:")
        print("    - verify_integrity() now accepts previous_record parameter")
        print("    - Checks that record.previous_hash == previous_record.entry_hash")
        print("    - verify_chain_integrity() validates entire chain sequentially")
        print("    - Any tampering with previous_hash is immediately detected")
        print()
        
        return True


if __name__ == "__main__":
    print()
    success = demonstrate_hash_chain_protection()
    print()
    
    if success:
        print("=" * 70)
        print("✅ DEMONSTRATION COMPLETE: Hash chain integrity protection works!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("=" * 70)
        print("❌ DEMONSTRATION FAILED: Hash chain protection not working!")
        print("=" * 70)
        sys.exit(1)
