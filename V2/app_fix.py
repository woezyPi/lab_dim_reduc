def render_advanced_results_panel(X, X_processed, embeddings_dict, y, class_names, meta):
    """Render the advanced results panel with all features."""
    st.markdown("---")

    # Store embeddings and data in session state for persistence
    if embeddings_dict:
        st.session_state.embeddings_dict = embeddings_dict
        st.session_state.X = X
        st.session_state.X_processed = X_processed
        st.session_state.y = y
        st.session_state.class_names = class_names
        st.session_state.meta = meta

    with st.expander("🔎 Résultats avancés", expanded=False):
        st.markdown("### Analyses approfondies des embeddings")

        # Check if we have embeddings available
        if 'embeddings_dict' not in st.session_state or not st.session_state.embeddings_dict:
            st.warning("⚠️ Aucun embedding disponible. Veuillez d'abord générer des embeddings.")
            return

        # Retrieve data from session state
        embeddings_dict = st.session_state.embeddings_dict
        X = st.session_state.X
        X_processed = st.session_state.X_processed
        y = st.session_state.y
        class_names = st.session_state.class_names
        meta = st.session_state.meta

        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Visualizations 3D",
            "📈 Trustworthiness vs k",
            "🎯 Matrices de Confusion KNN",
            "💾 Export CSV",
            "🏔️ Relief 3D (Digits)"
        ])

        # Tab 1: 3D Visualizations
        with tab1:
            st.markdown("#### Visualisations 3D des embeddings")

            col1, col2 = st.columns([1, 2])
            with col1:
                viz_method = st.selectbox(
                    "Méthode 3D:",
                    ["PCA", "UMAP", "t-SNE"],
                    key="3d_viz_method"
                )

                if st.button("Générer visualisation 3D", key="generate_3d"):
                    with st.spinner(f"Génération de la visualisation 3D {viz_method}..."):
                        try:
                            fig_3d = create_3d_visualization(X_processed, y, class_names, viz_method)
                            st.session_state.fig_3d = fig_3d
                            st.success("✅ Visualisation 3D générée avec succès!")
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la génération: {str(e)}")

            with col2:
                if 'fig_3d' in st.session_state:
                    st.plotly_chart(st.session_state.fig_3d, use_container_width=True)

        # Tab 2: Trustworthiness vs k curves
        with tab2:
            st.markdown("#### Courbes Trustworthiness vs k")

            col1, col2 = st.columns([1, 2])
            with col1:
                max_k = st.slider("Valeur k maximale", 5, 100, 50, key="max_k_trust")

                if st.button("Calculer courbes", key="calc_trust_curves") and embeddings_dict:
                    with st.spinner("Calcul des courbes de trustworthiness..."):
                        try:
                            fig_trust = plot_trustworthiness_vs_k(X, embeddings_dict, y, max_k)
                            st.session_state.fig_trust = fig_trust
                            st.success("✅ Courbes de trustworthiness calculées!")
                        except Exception as e:
                            st.error(f"❌ Erreur lors du calcul: {str(e)}")

            with col2:
                if 'fig_trust' in st.session_state:
                    st.pyplot(st.session_state.fig_trust)

        # Tab 3: KNN Confusion matrices
        with tab3:
            st.markdown("#### Matrices de Confusion KNN sur les Embeddings")

            if embeddings_dict:
                col1, col2 = st.columns([1, 2])
                with col1:
                    selected_method = st.selectbox(
                        "Méthode d'embedding:",
                        list(embeddings_dict.keys()),
                        key="knn_method"
                    )
                    k_neighbors = st.slider("Nombre de voisins (k)", 1, 20, 5, key="knn_k")

                    if st.button("Générer matrice", key="generate_knn_matrix"):
                        with st.spinner(f"Génération de la matrice de confusion KNN..."):
                            try:
                                X_emb = embeddings_dict[selected_method]
                                fig_cm, accuracy = create_knn_confusion_matrix(X_emb, y, class_names, k_neighbors)
                                st.session_state.fig_cm = fig_cm
                                st.session_state.knn_accuracy = accuracy
                                st.success(f"✅ Matrice générée - Précision: {accuracy:.2%}")
                            except Exception as e:
                                st.error(f"❌ Erreur lors de la génération: {str(e)}")

                with col2:
                    if 'fig_cm' in st.session_state:
                        st.pyplot(st.session_state.fig_cm)
                        if 'knn_accuracy' in st.session_state:
                            st.metric("Accuracy", f"{st.session_state.knn_accuracy:.2%}")
            else:
                st.info("Générez d'abord des embeddings pour voir les matrices de confusion.")

        # Tab 4: CSV Export
        with tab4:
            st.markdown("#### Export CSV des Embeddings")

            if embeddings_dict:
                st.info("Exporter tous les embeddings calculés dans un fichier CSV.")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Générer CSV", key="generate_csv"):
                        try:
                            csv_content = export_embeddings_csv(embeddings_dict, y, class_names)
                            st.session_state.csv_content = csv_content
                            st.success("✅ CSV généré avec succès!")
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la génération CSV: {str(e)}")

                with col2:
                    if 'csv_content' in st.session_state:
                        st.download_button(
                            label="📥 Télécharger CSV",
                            data=st.session_state.csv_content,
                            file_name=f"embeddings_{meta.get('name', 'dataset')}.csv",
                            mime="text/csv"
                        )

                # Preview of CSV structure
                if 'csv_content' in st.session_state:
                    st.markdown("**Aperçu du fichier CSV:**")
                    df_preview = pd.read_csv(io.StringIO(st.session_state.csv_content))
                    st.dataframe(df_preview.head(10), use_container_width=True)
            else:
                st.info("Générez d'abord des embeddings pour pouvoir les exporter.")

        # Tab 5: 3D Relief for Digits
        with tab5:
            st.markdown("#### Visualisation en Relief 3D (Dataset Digits)")

            col1, col2 = st.columns([1, 2])
            with col1:
                # Check if this is the Digits dataset
                if meta and meta.get('name', '').lower() in ['digits', 'digits (mnist)', 'mnist']:
                    selected_digit = st.selectbox(
                        "Choisir le chiffre:",
                        list(range(10)),
                        format_func=lambda x: f"Chiffre {x}",
                        key="relief_digit"
                    )

                    if st.button("Générer relief 3D", key="generate_relief"):
                        with st.spinner("Génération du relief 3D..."):
                            try:
                                fig_relief, error = create_digits_3d_relief(X, y, class_names, selected_digit)
                                if fig_relief is not None:
                                    st.session_state.fig_relief = fig_relief
                                    st.success("✅ Relief 3D généré avec succès!")
                                else:
                                    st.error(error)
                            except Exception as e:
                                st.error(f"❌ Erreur lors de la génération: {str(e)}")
                else:
                    st.warning("Cette visualisation est optimisée pour le dataset Digits.")

            with col2:
                if 'fig_relief' in st.session_state:
                    st.plotly_chart(st.session_state.fig_relief, use_container_width=True)

    return None