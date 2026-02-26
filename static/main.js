/* ZTF–Gaia Cross-Match — Interactive Table */

let allCandidates = [];
let filteredCandidates = [];
let currentPage = 1;
let pageSize = 50;
let sortColumn = "score";
let sortAsc = false;
let activeCategory = null;

function init() {
    const dataEl = document.getElementById("candidate-data");
    if (!dataEl) return;
    allCandidates = JSON.parse(dataEl.textContent);
    filteredCandidates = [...allCandidates];
    sortData();

    setupStatCards();
    setupSortHeaders();
    renderTable();
    renderPagination();
}

/* ── Category card filtering ──────────────────────────────────────── */

function setupStatCards() {
    document.querySelectorAll(".stat-card[data-category]").forEach(function (card) {
        card.addEventListener("click", function () {
            var cat = card.dataset.category;
            if (activeCategory === cat) {
                activeCategory = null;
                card.classList.remove("active");
            } else {
                document.querySelectorAll(".stat-card").forEach(function (c) {
                    c.classList.remove("active");
                });
                activeCategory = cat;
                card.classList.add("active");
            }
            applyFilter();
        });
    });
}

function applyFilter() {
    if (activeCategory) {
        filteredCandidates = allCandidates.filter(function (c) {
            return c.category === activeCategory;
        });
    } else {
        filteredCandidates = allCandidates.slice();
    }
    currentPage = 1;
    sortData();
    renderTable();
    renderPagination();
}

/* ── Column sorting ───────────────────────────────────────────────── */

function setupSortHeaders() {
    document.querySelectorAll("th[data-sort]").forEach(function (th) {
        th.addEventListener("click", function () {
            var col = th.dataset.sort;
            if (sortColumn === col) {
                sortAsc = !sortAsc;
            } else {
                sortColumn = col;
                sortAsc = (col === "rank");
            }
            document.querySelectorAll("th[data-sort]").forEach(function (h) {
                h.classList.remove("sort-asc", "sort-desc");
            });
            th.classList.add(sortAsc ? "sort-asc" : "sort-desc");
            sortData();
            currentPage = 1;
            renderTable();
            renderPagination();
        });
    });

    // Mark the default sort column
    var defaultTh = document.querySelector('th[data-sort="score"]');
    if (defaultTh) defaultTh.classList.add("sort-desc");
}

function sortData() {
    filteredCandidates.sort(function (a, b) {
        var va = a[sortColumn];
        var vb = b[sortColumn];
        if (va == null) va = sortAsc ? Infinity : -Infinity;
        if (vb == null) vb = sortAsc ? Infinity : -Infinity;
        if (typeof va === "string") {
            return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
        }
        return sortAsc ? va - vb : vb - va;
    });
}

/* ── Table rendering ──────────────────────────────────────────────── */

function renderTable() {
    var tbody = document.getElementById("candidates-tbody");
    if (!tbody) return;

    var start = (currentPage - 1) * pageSize;
    var page = filteredCandidates.slice(start, start + pageSize);

    var html = "";
    for (var i = 0; i < page.length; i++) {
        var c = page[i];
        var catLower = c.category.toLowerCase();
        var amp = c.amplitude != null ? c.amplitude.toFixed(3) : "\u2014";
        var nobs = c.nobs != null ? Math.round(c.nobs) : "\u2014";
        var gmag = c.gaia_g_mag != null ? c.gaia_g_mag.toFixed(1) : "\u2014";
        var surveys = c.survey_matches || "";

        html += '<tr class="cat-' + catLower + '-row">'
            + "<td>" + (start + i + 1) + "</td>"
            + '<td><a href="sources/' + c.id + '.html">' + c.id + "</a></td>"
            + "<td>" + c.ra.toFixed(5) + "</td>"
            + "<td>" + c.dec.toFixed(5) + "</td>"
            + '<td><span class="cat-badge cat-' + catLower + '">' + c.category + "</span></td>"
            + "<td>" + c.score.toFixed(3) + "</td>"
            + "<td>" + amp + "</td>"
            + "<td>" + nobs + "</td>"
            + "<td>" + gmag + "</td>"
            + "<td>" + surveys + "</td>"
            + "</tr>";
    }
    tbody.innerHTML = html;
}

/* ── Pagination ───────────────────────────────────────────────────── */

function renderPagination() {
    var pag = document.getElementById("pagination");
    var pagBottom = document.getElementById("pagination-bottom");
    if (!pag) return;

    var totalPages = Math.max(1, Math.ceil(filteredCandidates.length / pageSize));
    var catLabel = activeCategory ? " (Category " + activeCategory + ")" : "";
    var showing = filteredCandidates.length + " candidates" + catLabel;

    var html = '<span class="page-info">Showing ' + showing
        + " \u2014 Page " + currentPage + " of " + totalPages + "</span>"
        + '<div class="page-buttons">';

    if (currentPage > 1) {
        html += '<button onclick="goToPage(1)">\u00ab First</button>';
        html += '<button onclick="goToPage(' + (currentPage - 1) + ')">\u2039 Prev</button>';
    }

    var maxButtons = 5;
    var startPage = Math.max(1, currentPage - Math.floor(maxButtons / 2));
    var endPage = Math.min(totalPages, startPage + maxButtons - 1);
    startPage = Math.max(1, endPage - maxButtons + 1);

    for (var p = startPage; p <= endPage; p++) {
        var cls = p === currentPage ? ' class="active"' : "";
        html += "<button" + cls + ' onclick="goToPage(' + p + ')">' + p + "</button>";
    }

    if (currentPage < totalPages) {
        html += '<button onclick="goToPage(' + (currentPage + 1) + ')">Next \u203a</button>';
        html += '<button onclick="goToPage(' + totalPages + ')">Last \u00bb</button>';
    }

    html += "</div>";

    // Page size selector
    html += '<span class="page-size-select">Per page: '
        + '<select onchange="changePageSize(this.value)">';
    [25, 50, 100, 200].forEach(function (n) {
        var sel = n === pageSize ? " selected" : "";
        html += "<option value=" + n + sel + ">" + n + "</option>";
    });
    html += "</select></span>";

    pag.innerHTML = html;
    if (pagBottom) pagBottom.innerHTML = html;
}

function goToPage(page) {
    currentPage = page;
    renderTable();
    renderPagination();
    var table = document.getElementById("candidates-table");
    if (table) table.scrollIntoView({ behavior: "smooth" });
}

function changePageSize(val) {
    pageSize = parseInt(val, 10);
    currentPage = 1;
    renderTable();
    renderPagination();
}

document.addEventListener("DOMContentLoaded", init);
