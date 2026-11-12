# setup_gtk4.jl — one-time environment update for the Gtk4 branch
#
#   julia setup_gtk4.jl
#
# The Manifest.toml from the GTK 3 version still lists GtkObservables 1.x (which needs Gtk).
# `Pkg.resolve()` keeps every version in the Manifest, so it cannot move to GtkObservables 2
# ("empty intersection between GtkObservables@1.2.9 and project compatibility 2").
# Here only GtkObservables is upgraded and Gtk4 is added; all other packages keep their
# current versions (CairoMakie etc. are not touched).
# Commit the updated Manifest.toml afterwards so others get the same versions.

using Pkg
Pkg.activate(@__DIR__)

println("Upgrading GtkObservables 1.x -> 2.x and adding Gtk4 (other packages unchanged)...")
# default preserve (PRESERVE_TIERED): first tries to keep every other version as it is
try
    Pkg.add([PackageSpec(name="Gtk4"),
             PackageSpec(name="GtkObservables", version="2")])
    Pkg.instantiate()
catch e
    showerror(stderr, e)
    println(stderr, """


    The targeted upgrade failed. Fallback (re-resolves ALL packages to current compatible versions):
      1. delete Manifest.toml in this folder
      2. julia --project=. -e "using Pkg; Pkg.instantiate()"
    """)
    exit(1)
end

println("\nDone. Gtk is no longer needed by this environment.")
println("Start the GUI with:  julia --project=. -t auto Jwi-eak.jl")
