# Changelog

- 2026-01-24: Added `.context/` project memory files for plan, architecture, changelog, and current task tracking.
- 2026-01-24: Expanded `.context/CURRENT_TASK.md` with command-level runbook steps.
- 2026-01-24: Added diffusers compatibility shims (torch.xpu and pytree) to backup GUI.
- 2026-01-24: Wrapped pytree register helper to ignore unsupported kwargs on torch 2.1.
- 2026-01-24: Clear GT image on new input load in both GUI files.
- 2026-01-24: Added force_output_refresh to backup GUI to reduce blank output rendering.
- 2026-01-24: Guarded GUI image refresh with TclError handling to avoid crashes on X11.
- 2026-01-24: Added output render fallback when zoomed render fails in both GUI files.
- 2026-01-24: Avoid clearing output image during force refresh to prevent blank frames.
- 2026-01-24: Always reapply output image via show_image_ctk in force refresh.
