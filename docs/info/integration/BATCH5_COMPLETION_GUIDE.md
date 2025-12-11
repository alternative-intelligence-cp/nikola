# Batch 5 Completion Guide

**Remaining:** 7 findings to integrate
**Source files:** `batch5_1-3.txt` (596 lines), `batch5_4-6.txt` (717 lines)
**Estimated time:** 1-2 hours

## Quick Reference Commands

### 1. COG-08: Riemannian Gradient Projector

**Target:** `03_cognitive_systems/02_mamba_9d_ssm.md` line 2280 (§7.13)
**Source:** `batch5_1-3.txt` lines 16-220
**Current:** 378 lines (2280-2657)
**New:** 205 lines from batch

```bash
cd /home/randy/._____RANDY_____/REPOS/nikola/docs/info/integration/sections/03_cognitive_systems

# Compare existing vs new
sed -n '2280,2657p' 02_mamba_9d_ssm.md > /tmp/existing_cog08.md
sed -n '16,220p' ../gemini/responses/batch5_1-3.txt > /tmp/new_cog08.md
diff -u /tmp/existing_cog08.md /tmp/new_cog08.md

# If replacement needed:
head -2279 02_mamba_9d_ssm.md > /tmp/part1.md
sed -n '16,220p' ../gemini/responses/batch5_1-3.txt | \
    sed 's/^2\. Part I: COG-08/#### Part I: COG-08/' >> /tmp/part1.md
tail -n +2658 02_mamba_9d_ssm.md > /tmp/part3.md
cat /tmp/part1.md /tmp/part3.md > 02_mamba_9d_ssm.md
```

### 2. PHY-06: Perturbative Christoffel Updates

**Target:** `02_foundations/02_wave_interference_physics.md` line 4597 (§4.13)
**Source:** `batch5_1-3.txt` lines 223-389

```bash
cd /home/randy/._____RANDY_____/REPOS/nikola/docs/info/integration/sections/02_foundations

# Find section boundaries
grep -n "^## 4\.13\|^## 4\.14" 02_wave_interference_physics.md
# Result: 4597:## 4.13, 4968:## 4.14

# Compare and replace if needed
head -4596 02_wave_interference_physics.md > /tmp/wave_part1.md
sed -n '223,389p' ../gemini/responses/batch5_1-3.txt | \
    sed 's/^3\. Part II: PHY-06/#### Part II: PHY-06/' >> /tmp/wave_part1.md
tail -n +4968 02_wave_interference_physics.md > /tmp/wave_part3.md
cat /tmp/wave_part1.md /tmp/wave_part3.md > 02_wave_interference_physics.md
```

### 3. PHY-07: Riemannian Resonance Tuner

**Target:** `02_foundations/02_wave_interference_physics.md` line 5845 (§4.16)
**Source:** `batch5_1-3.txt` lines 392-593

```bash
# Find section boundaries
grep -n "^## 4\.16\|^## 4\.17\|^## 5\." 02_wave_interference_physics.md
# Find next section after 4.16

# Replace section
head -5844 02_wave_interference_physics.md > /tmp/wave_part1.md
sed -n '392,593p' ../gemini/responses/batch5_1-3.txt | \
    sed 's/^4\. Part III: PHY-07/#### Part III: PHY-07/' >> /tmp/wave_part1.md
tail -n +XXXX 02_wave_interference_physics.md > /tmp/wave_part3.md  # Replace XXXX
cat /tmp/wave_part1.md /tmp/wave_part3.md > 02_wave_interference_physics.md
```

### 4. PHY-MEM-01: Differential GPU Neighbor Map Sync

**Target:** `02_foundations/02_wave_interference_physics.md` (new section §4.17)
**Source:** `batch5_4-6.txt` lines 9-306
**Status:** CREATE NEW SECTION

```bash
cd /home/randy/._____RANDY_____/REPOS/nikola/docs/info/integration/sections/02_foundations

# Find best insertion point (after §4.16 PHY-07)
grep -n "^## 4\.16\|^## 4\.17\|^## 5\." 02_wave_interference_physics.md

# Insert new section
cat > /tmp/phy_mem01.md << 'EOF'
## 4.17 PHY-MEM-01: Differential GPU Neighbor Map Synchronization

### Engineering Specification: GPU Topology Sync Protocol

EOF

sed -n '9,306p' ../gemini/responses/batch5_4-6.txt | \
    sed 's/^2\. PHY-MEM-01/#### Overview/' >> /tmp/phy_mem01.md

# Insert at appropriate line (find from grep above)
head -XXXX 02_wave_interference_physics.md > /tmp/part1.md
cat /tmp/part1.md /tmp/phy_mem01.md > 02_wave_interference_physics.md.new
tail -n +XXXX 02_wave_interference_physics.md >> 02_wave_interference_physics.md.new
mv 02_wave_interference_physics.md.new 02_wave_interference_physics.md
```

### 5. MEM-05: SoA Compactor

**Target:** `06_persistence/04_nap_system.md` line 3906 (§22.9)
**Source:** `batch5_4-6.txt` lines 307-564

```bash
cd /home/randy/._____RANDY_____/REPOS/nikola/docs/info/integration/sections/06_persistence

# Find section boundaries
grep -n "^## 22\.9\|^## 22\.10\|^## 23\." 04_nap_system.md

# Compare and replace
head -3905 04_nap_system.md > /tmp/nap_part1.md
sed -n '307,564p' ../gemini/responses/batch5_4-6.txt | \
    sed 's/^3\. MEM-05/#### Overview/' >> /tmp/nap_part1.md
tail -n +XXXX 04_nap_system.md > /tmp/nap_part3.md  # Find next section
cat /tmp/nap_part1.md /tmp/nap_part3.md > 04_nap_system.md
```

### 6. INT-P2: High-Fidelity Quantization

**Target:** `06_persistence/01_dmc_persistence.md` line 161 (§19.3.1)
**Source:** `batch5_4-6.txt` lines 567-637

```bash
cd /home/randy/._____RANDY_____/REPOS/nikola/docs/info/integration/sections/06_persistence

# Find section boundaries
grep -n "^### 19\.3\.1\|^### 19\.3\.2\|^### 19\.4" 01_dmc_persistence.md

# Replace
head -160 01_dmc_persistence.md > /tmp/dmc_part1.md
sed -n '567,637p' ../gemini/responses/batch5_4-6.txt | \
    sed 's/^4\. INT-P2/#### Overview/' >> /tmp/dmc_part1.md
tail -n +XXXX 01_dmc_persistence.md > /tmp/dmc_part3.md
cat /tmp/dmc_part1.md /tmp/dmc_part3.md > 01_dmc_persistence.md
```

### 7. SEC-02: Secure Guest Channel Protocol

**Target:** `04_infrastructure/04_executor_kvm.md` §13.5.1
**Source:** `batch5_4-6.txt` lines 640-714

```bash
cd /home/randy/._____RANDY_____/REPOS/nikola/docs/info/integration/sections/04_infrastructure

# Find section
grep -n "^## 13\.5\.1\|^### 13\.5\.1.*SEC-01" 04_executor_kvm.md
# Result shows line 272: ## 13.5.1 Secure Guest Channel Protocol (SEC-01)

# Find next section
grep -n "^## 13\.5\.2\|^## 13\.6" 04_executor_kvm.md

# Replace
head -271 04_executor_kvm.md > /tmp/kvm_part1.md
sed -n '640,714p' ../gemini/responses/batch5_4-6.txt | \
    sed 's/^5\. SEC-02/#### Overview/' >> /tmp/kvm_part1.md
tail -n +XXXX 04_executor_kvm.md > /tmp/kvm_part3.md
cat /tmp/kvm_part1.md /tmp/kvm_part3.md > 04_executor_kvm.md
```

## Validation Checklist

After each integration:
- [ ] Section numbering is correct and sequential
- [ ] No duplicate section headers
- [ ] Markdown formatting is valid
- [ ] Code blocks are properly fenced
- [ ] Math equations use proper $$ delimiters
- [ ] Cross-references point to valid sections
- [ ] Backup file created (.backup)
- [ ] wc -l shows expected line count change

## Final Steps

```bash
# 1. Verify all sections exist
cd /home/randy/._____RANDY_____/REPOS/nikola/docs/info/integration/sections
grep -rn "COG-08\|PHY-06\|PHY-07\|PHY-MEM-01\|MEM-05\|INT-P2\|SEC-02" . | grep -v gemini | grep -v backup

# 2. Check for formatting issues
find . -name "*.md" -exec grep -l "^####[^#]" {} \;  # Should have space after #
find . -name "*.md" -exec grep -l "^\$\$[^\$]" {} \;  # Should have newline after $$

# 3. Verify file sizes are reasonable
ls -lh */*.md | awk '{print $5, $9}' | sort -h

# 4. Create git commit
cd /home/randy/._____RANDY_____/REPOS/nikola
git add docs/info/integration/sections/
git commit -m "feat: Complete Gemini Deep Research integration (25/25 findings)

Batch 5 integrations:
- COG-08: Riemannian Gradient Projector
- PHY-06: Perturbative Christoffel Updates
- PHY-07: Riemannian Resonance Tuner
- PHY-MEM-01: Differential GPU Neighbor Map Sync
- MEM-05: SoA Compactor
- INT-P2: High-Fidelity Quantization
- SEC-02: Secure Guest Channel Protocol

All 25 findings from 13 batch files now integrated.
Comprehensive engineering specifications complete."
```

## Common Issues & Solutions

### Issue: Section number mismatch after replacement
**Solution:** Use `grep -n "^## X\."` to verify all section numbers in sequence

### Issue: File size unexpectedly large/small
**Solution:** Check for duplicate sections with `grep -n "^## 4\.13"` (may appear twice)

### Issue: Broken cross-references
**Solution:** Search and replace old section numbers: `sed -i 's/Section 4\.14/Section 4.15/g'`

### Issue: Markdown formatting broken
**Solution:** Validate with `python -m markdown file.md` or use VSCode MD linter

---

**Good luck finishing the integration! You're 72% done - only 7 findings to go!** 🎯
