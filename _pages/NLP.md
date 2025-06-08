---
layout: default
title: NLP
permalink: /NLP/
parent: Category
---
<ul>
  {% assign cat = site.categories.NLP %}
  {% for post in cat %}
    <li><a href="{{ post.url }}">{{ post.title }}</a> ({{ post.date | date: "%Y-%m-%d" }})</li>
  {% endfor %}
</ul>