BEGIN { mode=0 }
/<body>/ { sub(/.*<body>/,""); mode=1 }
/<\/body>/ { sub(/<\/body>.*/,""); print; mode=0 }
{ if(mode) print }
